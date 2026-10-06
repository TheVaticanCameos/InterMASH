import torch
from torch import Tensor
from typing import Optional, Union, Tuple
from mash_hoi.source.Utils.utils import KeyLUT, gray2binary, binary2gray


class ZOrderSerializer:
    def __init__(self) -> None:
        self.key_lut = KeyLUT()
    
    def encode(
        self,
        grid_coord: Tensor,  # shape: [N, 3]
        depth: int = 16
    ) -> Tensor:
        x, y, z = grid_coord[:,0].long(), grid_coord[:,1].long(), grid_coord[:,2].long()
        code = self.xyz2key(x, y, z, b=None, depth=depth)
        return code

    def decode(
        self,
        code: Tensor, 
        depth: int = 16
    ) -> Tensor:
        x, y, z, b = self.key2xyz(code, depth)
        grid_coord = torch.stack([x, y, z], dim=-1)
        return grid_coord
    
    def xyz2key(
        self,
        x: Tensor,  # shape: [N]
        y: Tensor,  # shape: [N]
        z: Tensor,  # shape: [N]
        b: Optional[Union[Tensor, int]] = None,
        depth: int = 16,
    ) -> Tensor:
        r"""Encodes :attr:`x`, :attr:`y`, :attr:`z` coordinates to the shuffled keys
        based on pre-computed look up tables. The speed of this function is much
        faster than the method based on for-loop.

        Args:
        x (torch.Tensor): The x coordinate.
        y (torch.Tensor): The y coordinate.
        z (torch.Tensor): The z coordinate.
        b (torch.Tensor or int): The batch index of the coordinates, and should be
            smaller than 32768. If :attr:`b` is :obj:`torch.Tensor`, the size of
            :attr:`b` must be the same as :attr:`x`, :attr:`y`, and :attr:`z`.
        depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
        """

        EX, EY, EZ = self.key_lut.encode_lut(x.device)
        x, y, z = x.long(), y.long(), z.long()

        mask = 255 if depth > 8 else (1 << depth) - 1
        key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
        if depth > 8:
            mask = (1 << (depth - 8)) - 1
            key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
            key = key16 << 24 | key

        if b is not None:
            b = b.long()
            key = b << 48 | key

        return key

    def key2xyz(
        self, 
        key: Tensor, 
        depth: int = 16
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Union[Tensor, int]]]:
        r"""Decodes the shuffled key to :attr:`x`, :attr:`y`, :attr:`z` coordinates
        and the batch index based on pre-computed look up tables.

        Args:
        key (torch.Tensor): The shuffled key.
        depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
        """

        DX, DY, DZ = self.key_lut.decode_lut(key.device)
        x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)

        b = key >> 48
        key = key & ((1 << 48) - 1)

        n = (depth + 2) // 3
        for i in range(n):
            k = key >> (i * 9) & 511
            x = x | (DX[k] << (i * 3))
            y = y | (DY[k] << (i * 3))
            z = z | (DZ[k] << (i * 3))

        return x, y, z, b


class HilbertSerializer:
    def __init__(self, num_dims: int = 3) -> None:
        self.num_dims = num_dims

    def encode(
        self, 
        grid_coord: Tensor,  # shape: [N, 3]
        depth: int = 16
    ) -> Tensor:
        num_dims = self.num_dims
        num_bits = depth

        # Keep around the original shape for later.
        orig_shape = grid_coord.shape
        bitpack_mask = 1 << torch.arange(0, 8).to(grid_coord.device)
        bitpack_mask_rev = bitpack_mask.flip(-1)

        if orig_shape[-1] != num_dims:
            raise ValueError(
                """
        The shape of locs was surprising in that the last dimension was of size
        %d, but num_dims=%d.  These need to be equal.
        """
                % (orig_shape[-1], num_dims)
            )

        if num_dims * num_bits > 63:
            raise ValueError(
                """
        num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
        into a int64.  Are you sure you need that many points on your Hilbert
        curve?
        """
                % (num_dims, num_bits, num_dims * num_bits)
            )

        # Treat the location integers as 64-bit unsigned and then split them up into
        # a sequence of uint8s.  Preserve the association by dimension.
        locs_uint8 = grid_coord.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)

        # Now turn these into bits and truncate to num_bits.
        gray = (
            locs_uint8.unsqueeze(-1)
            .bitwise_and(bitpack_mask_rev)
            .ne(0)
            .byte()
            .flatten(-2, -1)[..., -num_bits:]
        )

        # Run the decoding process the other way.
        # Iterate forwards through the bits.
        for bit in range(0, num_bits):
            # Iterate forwards through the dimensions.
            for dim in range(0, num_dims):
                # Identify which ones have this bit active.
                mask = gray[:, dim, bit]

                # Where this bit is on, invert the 0 dimension for lower bits.
                gray[:, 0, bit + 1 :] = torch.logical_xor(
                    gray[:, 0, bit + 1 :], mask[:, None]
                )

                # Where the bit is off, exchange the lower bits with the 0 dimension.
                to_flip = torch.logical_and(
                    torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                    torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
                )
                gray[:, dim, bit + 1 :] = torch.logical_xor(
                    gray[:, dim, bit + 1 :], to_flip
                )
                gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

        # Now flatten out.
        gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))

        # Convert Gray back to binary.
        hh_bin = gray2binary(gray)

        # Pad back out to 64 bits.
        extra_dims = 64 - num_bits * num_dims
        padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)

        # Convert binary values into uint8s.
        hh_uint8 = (
            (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
            .sum(2)
            .squeeze()
            .type(torch.uint8)
        )

        # Convert uint8s into uint64s.
        hh_uint64 = hh_uint8.view(torch.int64).squeeze()

        return hh_uint64

    def decode(
        self,
        code: Tensor,
        depth: int = 16
    ) -> Tensor:
        num_dims = self.num_dims
        num_bits = depth

        if num_dims * num_bits > 64:
            raise ValueError(
                """
        num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
        into a uint64.  Are you sure you need that many points on your Hilbert
        curve?
        """
                % (num_dims, num_bits)
            )

        # Handle the case where we got handed a naked integer.
        hilberts = torch.atleast_1d(code)

        # Keep around the shape for later.
        orig_shape = hilberts.shape
        bitpack_mask = 2 ** torch.arange(0, 8).to(hilberts.device)
        bitpack_mask_rev = bitpack_mask.flip(-1)

        # Treat each of the hilberts as a s equence of eight uint8.
        # This treats all of the inputs as uint64 and makes things uniform.
        hh_uint8 = (
            hilberts.ravel().type(torch.int64).view(torch.uint8).reshape((-1, 8)).flip(-1)
        )

        # Turn these lists of uints into lists of bits and then truncate to the size
        # we actually need for using Skilling's procedure.
        hh_bits = (
            hh_uint8.unsqueeze(-1)
            .bitwise_and(bitpack_mask_rev)
            .ne(0)
            .byte()
            .flatten(-2, -1)[:, -num_dims * num_bits :]
        )

        # Take the sequence of bits and Gray-code it.
        gray = binary2gray(hh_bits)

        # There has got to be a better way to do this.
        # I could index them differently, but the eventual packbits likes it this way.
        gray = gray.reshape((-1, num_bits, num_dims)).swapaxes(1, 2)

        # Iterate backwards through the bits.
        for bit in range(num_bits - 1, -1, -1):
            # Iterate backwards through the dimensions.
            for dim in range(num_dims - 1, -1, -1):
                # Identify which ones have this bit active.
                mask = gray[:, dim, bit]

                # Where this bit is on, invert the 0 dimension for lower bits.
                gray[:, 0, bit + 1 :] = torch.logical_xor(
                    gray[:, 0, bit + 1 :], mask[:, None]
                )

                # Where the bit is off, exchange the lower bits with the 0 dimension.
                to_flip = torch.logical_and(
                    torch.logical_not(mask[:, None]),
                    torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
                )
                gray[:, dim, bit + 1 :] = torch.logical_xor(
                    gray[:, dim, bit + 1 :], to_flip
                )
                gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

        # Pad back out to 64 bits.
        extra_dims = 64 - num_bits
        padded = torch.nn.functional.pad(gray, (extra_dims, 0), "constant", 0)

        # Now chop these up into blocks of 8.
        locs_chopped = padded.flip(-1).reshape((-1, num_dims, 8, 8))

        # Take those blocks and turn them unto uint8s.
        # from IPython import embed; embed()
        locs_uint8 = (locs_chopped * bitpack_mask).sum(3).squeeze().type(torch.uint8)

        # Finally, treat these as uint64s.
        flat_locs = locs_uint8.view(torch.int64)

        # Return them in the expected shape.
        return flat_locs.reshape((*orig_shape, num_dims))


class Serializer:
    def __init__(self) -> None:
        self.z_order_serializer = ZOrderSerializer()
        self.hilbert_serializer = HilbertSerializer()
    
    @torch.inference_mode()
    def encode(
        self,
        grid_coord: Tensor,
        batch: Optional[Tensor] = None,
        depth: int = 16,
        order: str = "z"
    ) -> Tensor:
        match order:
            case "z":
                code = self.z_order_serializer.encode(grid_coord, depth)
            case "z-trans":
                code = self.z_order_serializer.encode(grid_coord[:, [1, 0, 2]], depth)
            case "hilbert":
                code = self.hilbert_serializer.encode(grid_coord, depth)
            case "hilbert-trans":
                code = self.hilbert_serializer.encode(grid_coord[:, [1, 0, 2]], depth)
            case _:
                raise NotImplementedError(f"order {order} not implemented")
        if batch is not None:
            batch = batch.long()
            code = batch << depth * 3 | code
        return code
    
    @torch.inference_mode()
    def decode(
        self,
        code: Tensor,
        depth: int = 16,
        order: str = "z"
    ) -> Tuple[Tensor, Tensor]:
        batch = code >> depth * 3
        code = code & ((1 << depth * 3) - 1)
        match order:
            case "z":
                grid_coord = self.z_order_serializer.decode(code, depth)
            case "hilbert":
                grid_coord = self.hilbert_serializer.decode(code, depth)
            case _:
                raise NotImplementedError(f"order {order} not implemented")
        return grid_coord, batch





# # TEST
# if __name__ == "__main__":
#     serializer = Serializer()
#     grid_coord = torch.randint(0, 1024, (1000000, 3))
#     code = serializer.encode(grid_coord, order="z")
#     grid_coord_, batch_ = serializer.decode(code, order="z")
#     import pdb
#     pdb.set_trace()
#     print(grid_coord)
#     print(grid_coord_)
#     print((grid_coord == grid_coord_).all())
