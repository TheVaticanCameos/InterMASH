import os
import sys

from lightning import seed_everything


sys.path.append("../chamfer-distance")
sys.path.append("../data-convert")
sys.path.append("../diff-curvature")
sys.path.append("../mesh-graph-cut")
sys.path.append("../mesh-cut")
sys.path.append("../mesh-sample")
sys.path.append("../sdf-generate")

from ma_sh.Demo.pipeline_convertor import demo_convert_hoi_cmap

if __name__ == "__main__":
    seed_everything(42)
    name = "ycb+wood_block_0"
    hand_root = "/data2/yxz/DATASET/DRO-data/DRO-Original/shadow/train"
    hand_path = os.path.join(hand_root, f"shadow_{name}_hand.ply")
    object_path = os.path.join(hand_root, f"shadow_{name}_object.ply")
    cmap_path = os.path.join(hand_root, f"shadow_{name}_cmap.npy")

    hand_template_info_path = "/data2/yxz/DATASET/template_info/shadow_template.npy"
    object_template_info_path = f"/data2/yxz/DATASET/DRO-data/DRO-Original/shadow/object_template/{name.split('+', 1)[1].rsplit('_', 1)[0]}.npy"

    print("=" * 50)
    print(hand_path)
    print(object_path)
    print(cmap_path)
    print(hand_template_info_path)
    print(object_template_info_path)
    print("=" * 50)

    output_path = f"./test_hoi_{name}.npy"
    save_log_folder = None
    cuda_id = "0"

    demo_convert_hoi_cmap(
        hand_path=hand_path, 
        object_path=object_path, 
        cmap_path=cmap_path,
        output_path=output_path, 
        save_log_folder=save_log_folder, 
        device=f"cuda:{cuda_id}",
        anchor_num=256, 
        hand_template_info_path=hand_template_info_path,
        object_template_info_path=object_template_info_path,
        hand_type="shadow",
        initial_method="middle_plane",
        use_closed_form_cmap=False
    )
    
