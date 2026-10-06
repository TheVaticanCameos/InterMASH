import os
import argparse
from ma_sh.Demo.view import demo_view_hoi


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description="view mash")

    parser.add_argument("--mash_path", help="mash data path")
    parser.add_argument("--show_mash", type=bool, default=True)
    parser.add_argument("--show_line_hand", type=bool, default=False)
    parser.add_argument("--show_line_object", type=bool, default=False)
    parser.add_argument("--show_positions", type=bool, default=False)
    parser.add_argument("--target_name", type=str, default="object")
    parser.add_argument("--feat_name", type=str, default="ortho_poses")
    parser.add_argument("--show_cmap", type=bool, default=False)
    args = parser.parse_args()

    mash_path: str = args.mash_path
    show_line_hand: bool = args.show_line_hand
    show_line_object: bool = args.show_line_object
    show_mash: bool = args.show_mash
    show_positions: bool = args.show_positions
    target_name: str = args.target_name
    feat_name: str = args.feat_name
    show_cmap: bool = args.show_cmap

    if mash_path.endswith(".npy"):
        demo_view_hoi(
            mash_path, 
            show_mash, 
            show_line_hand,
            show_line_object,
            show_positions, 
            target_name, 
            feat_name,
            show_cmap=show_cmap
        )
    else:
        print("[ERROR][view::__main__]")
        print("\t mash data path not valid!")
        print("\t mash_data_path:", mash_path)
