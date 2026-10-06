import torch
from torch import Tensor
import spconv.pytorch as spconv
from mash_hoi.source.Utils.serialization import Serializer
from typing import List, Optional


class AnchorSet:
    def __init__(
        self, 
        positions: Tensor,    # shape: [B, N, 3] TODO: here using 3, could be 2
        hand_feat: Tensor,    # shape: [B, N, d_feat]
        object_feat: Tensor,  # shape: [B, N, d_feat]
        batch: Optional[Tensor],    # shape: [B * N]
        grid_size: float,           # 0.02 for example
    ) -> None:
        self.batch_size = batch[-1].tolist() + 1
        self.num = positions.shape[1]
        self.coord = positions.reshape(-1, positions.shape[-1])

        self.hand_feat = hand_feat
        self.object_feat = object_feat
        
        self.batch = batch
        self.grid_size = grid_size

        self.grid_coord = torch.div(
            self.coord - self.coord.min(0)[0],
            self.grid_size, rounding_mode="trunc"
        ).int()

        self.serializer = Serializer()

        self.serialized_code: Optional[Tensor] = None
        self.serialized_order: Optional[Tensor] = None
        self.serialized_inverse: Optional[Tensor] = None

        self.sparse_shape: Optional[List[int]] = None
        
        self.sparse_conv_hand: Optional[spconv.SparseConvTensor] = None
        self.sparse_conv_object: Optional[spconv.SparseConvTensor] = None
    
    def __len__(self) -> int:
        return self.num
    
    def serialization(
        self, 
        order: List[str], 
        depth: Optional[int],
        shuffle_order: bool
    ) -> None:
        if depth is None:
            depth = int(self.grid_coord.max() + 1).bit_length()

        # assert depth * 3 + len(self.offset).bit_length() <= 63, "Depth exceeds int64 limit"
        assert depth <= 16, "Depth must be <= 16"
        
        code = [
            self.serializer.encode(self.grid_coord, batch=self.batch, depth=depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        
        order_tensor = torch.argsort(code)
        inverse = torch.zeros_like(order_tensor).scatter_(
            dim=1,
            index=order_tensor,
            src=torch.arange(0, code.shape[1], device=order_tensor.device).repeat(code.shape[0], 1)
        )

        if shuffle_order:
            perm = torch.randperm(len(code))
            code = code[perm]
            order_tensor = order_tensor[perm]
            inverse = inverse[perm]
        
        # Check code for NaN values
        if torch.isnan(code).any():
            raise ValueError("serialized_code contains NaN values in code")
        # Check order_tensor for NaN values
        if torch.isnan(order_tensor).any():
            raise ValueError("serialized_order contains NaN values in order_tensor")
        # Check inverse for NaN values
        if torch.isnan(inverse).any():
            raise ValueError("serialized_inverse contains NaN values in inverse")
        
        self.serialized_code = code
        self.serialized_order = order_tensor
        self.serialized_inverse = inverse
    
    def sparsify(
        self,
        pad: int = 96
    ) -> None:
        sparse_shape = torch.add(torch.max(self.grid_coord, dim=0).values, pad).tolist()
        indices = torch.cat(
            [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous()
        sparse_conv_hand = spconv.SparseConvTensor(
            features=self.hand_feat.view(-1, self.hand_feat.shape[-1]),
            indices=indices,
            spatial_shape=sparse_shape,
            batch_size=self.batch_size
        )
        sparse_conv_object = spconv.SparseConvTensor(
            features=self.object_feat.view(-1, self.object_feat.shape[-1]),
            indices=indices,
            spatial_shape=sparse_shape,
            batch_size=self.batch_size
        )

        self.sparse_shape = sparse_shape

        self.sparse_conv_hand = sparse_conv_hand
        self.sparse_conv_object = sparse_conv_object

# # TEST 
# if __name__ == "__main__":
#     batch_size = 4
#     point_num = 5
#     positions = [torch.rand(point_num, 3) for _ in range(batch_size)]
#     hand_feat = [torch.rand(point_num, 10) for _ in range(batch_size)]
#     object_feat = [torch.rand(point_num, 10) for _ in range(batch_size)]

#     batch = torch.arange(batch_size).repeat_interleave(point_num)
#     grid_size = 0.001

#     anchor_set = AnchorSet(
#         positions=positions, 
#         hand_feat=hand_feat,
#         object_feat=object_feat,
#         batch=batch,
#         grid_size=grid_size,
#         device="cuda:0"
#     )
#     anchor_set.serialization(["z", "z-trans", "hilbert", "hilbert-trans"], 10, True)
#     anchor_set.sparsify()

