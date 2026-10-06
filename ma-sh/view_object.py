import os
import argparse
from ma_sh.Demo.view import demo_view_object


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description="view mash")

    parser.add_argument("--mash_path", help="mash data path")
    parser.add_argument("--show_line_object", type=bool, default=False)
    parser.add_argument("--show_positions", type=bool, default=True)
    args = parser.parse_args()

    mash_path: str = args.mash_path
    show_line_object: bool = args.show_line_object
    show_positions: bool = args.show_positions

    if mash_path.endswith(".npy"):
        demo_view_object(
            mash_file_path=mash_path, 
            show_positions=show_positions, 
            show_object_line=show_line_object,
        )
    else:
        print("[ERROR][view::__main__]")
        print("\t mash data path not valid!")
        print("\t mash_data_path:", mash_path)
