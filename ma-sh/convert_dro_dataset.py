import os


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_num", type=int, default=256)
    parser.add_argument("--data_root", type=str, default="/data2/yxz/DATASET/DRO-data/DRO-Original")
    parser.add_argument("--template_root", type=str, default="/data2/yxz/DATASET/template_info")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--cuda_id", type=str, default="0")
    parser.add_argument("--hand_type", type=str, default="shadow")
    parser.add_argument("--output_root", type=str, default="/data2/yxz/DATASET/DRO-data/DRO-HOI-Dataset")
    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--current_num", type=int, default=0)
    parser.add_argument("--initial_method", type=str, default="middle_plane")
    args = parser.parse_args()

    anchor_num = args.anchor_num
    data_root = args.data_root
    template_root = args.template_root
    mode = args.mode
    cuda_id = args.cuda_id
    hand_type = args.hand_type
    output_root = args.output_root
    part_num = args.part_num
    current_num = args.current_num
    initial_method = args.initial_method

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    os.makedirs(os.path.join(output_root, hand_type, mode), exist_ok=True)

    import sys

    sys.path.append("../chamfer-distance")
    sys.path.append("../data-convert")
    sys.path.append("../diff-curvature")
    sys.path.append("../mesh-graph-cut")
    sys.path.append("../mesh-cut")
    sys.path.append("../mesh-sample")
    sys.path.append("../sdf-generate")

    from ma_sh.Demo.pipeline_convertor import demo_convert_hoi_cmap

    template_path = os.path.join(template_root, f"{hand_type}_template.npy")
    read_path = os.path.join(data_root, hand_type, mode)

    hand_files = []
    object_files = []
    cmap_files = []
    for root, dirs, files in os.walk(read_path):
        for file in files:
            assert file.startswith(hand_type)
            if file.endswith("_hand.ply"):
                hand_files.append(os.path.join(root, file))
            elif file.endswith("_object.ply"):
                object_files.append(os.path.join(root, file))
            elif file.endswith("_cmap.npy"):
                cmap_files.append(os.path.join(root, file))
    
    hand_files.sort()
    object_files.sort()
    cmap_files.sort()

    assert len(hand_files) == len(object_files) == len(cmap_files)
    total_length = len(hand_files)

    if part_num < 1:
        raise ValueError("part_num must be >= 1")
    if current_num < 0 or current_num >= part_num:
        raise ValueError("current_num must be in [0, part_num-1]")

    base = total_length // part_num
    rem = total_length % part_num
    part_idx = current_num
    start_idx = part_idx * base + (part_idx if part_idx < rem else rem)
    size = base + (1 if part_idx < rem else 0)
    end_idx = start_idx + size

    for idx in range(start_idx, end_idx):
        name = hand_files[idx].rsplit("_", 1)[0].split("/")[-1]
        output_dir = os.path.join(output_root, hand_type, mode, f"{name}_hoi.npy")
        
        if not os.path.exists(output_dir):
            try:
                demo_convert_hoi_cmap(
                    hand_path=hand_files[idx], 
                    object_path=object_files[idx], 
                    cmap_path=cmap_files[idx], 
                    output_path=output_dir, 
                    save_log_folder=None, 
                    anchor_num=anchor_num, 
                    template_info_path=template_path,
                    hand_type=hand_type,
                    initial_method=initial_method,
                    use_closed_form_cmap=True,
                )
            except Exception as e:
                print(f"[ERROR][Convertor::convertData] {name} failed: {e}")

