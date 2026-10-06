import os


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_num", type=int, default=256)
    parser.add_argument("--original_root", type=str, default="/data2/yxz/DATASET/GrabDataset")
    parser.add_argument("--target_root", type=str, default="/data2/yxz/DATASET/GrabDataset")
    parser.add_argument("--cuda_id", type=str, default="0")
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    
    anchor_num = args.anchor_num
    original_root = args.original_root
    target_root = args.target_root
    cuda_id = args.cuda_id
    mode = args.mode

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

    import sys

    sys.path.append("../chamfer-distance")
    sys.path.append("../data-convert")
    sys.path.append("../diff-curvature")
    sys.path.append("../mesh-graph-cut")
    sys.path.append("../mesh-cut")
    sys.path.append("../mesh-sample")
    sys.path.append("../sdf-generate")

    from ma_sh.Demo.pipeline_convertor import demo_convert_hoi

    template_info_path = os.path.join(original_root, "hand_template.npy")
    
    hand_files = []
    object_files = []
    for root, dirs, files in os.walk(os.path.join(original_root, mode)):
        for file in files:
            if file.endswith('_hand.ply'):
                hand_files.append(os.path.join(root, file))
            elif file.endswith('_object.ply'):
                object_files.append(os.path.join(root, file))

    hand_files.sort()
    object_files.sort()

    assert len(hand_files) == len(object_files)
    for i in range(len(hand_files)):
        assert hand_files[i].split('_')[:-1] == object_files[i].split('_')[:-1]

    total_length = len(hand_files)

    for idx in range(total_length):
        output_dir = os.path.join(target_root, mode, hand_files[idx].split('/')[-1][:-8] + "hoi.npy")
        
        # Check whether the output file exists; otherwise, call demo_convert_hoi
        if not os.path.exists(output_dir):
            try:
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                demo_convert_hoi(
                    hand_path=hand_files[idx],
                    object_path=object_files[idx],
                    output_path=output_dir,
                    save_log_folder=None,
                    anchor_num=anchor_num,
                    template_info_path=template_info_path
                )
            except Exception as e:
                print(f"处理文件 {hand_files[idx]} 和 {object_files[idx]} 时出错: {str(e)}")
