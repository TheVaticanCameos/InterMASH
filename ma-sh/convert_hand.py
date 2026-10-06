import os
import sys

sys.path.append("../chamfer-distance")
sys.path.append("../data-convert")
sys.path.append("../diff-curvature")
sys.path.append("../mesh-graph-cut")
sys.path.append("../mesh-cut")
sys.path.append("../mesh-sample")
sys.path.append("../sdf-generate")

from ma_sh.Demo.pipeline_convertor import demo_convert_hand

if __name__ == "__main__":

    hand_type = "shadow"

    template_info_path = f"/data2/yxz/DATASET/template_info/sphere_{hand_type}_template.npy"
    hand_path = f"/data2/yxz/DATASET/DRO-data/DRO-Original/{hand_type}/{hand_type}_template.ply"
    output_path = f"/data2/yxz/DATASET/DRO-data/DRO-HOI-Dataset/{hand_type}/mash_sphere_{hand_type}_template.npy"
    
    save_log_folder = "/data/yxz/MASH-HOI/MASH/ma-sh/test/log"
    cuda_id = "0"

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

    demo_convert_hand(
        hand_path=hand_path,
        output_path=output_path,
        save_log_folder=save_log_folder,
        anchor_num=256,
        device="cuda:0",
        template_info_path=template_info_path,
        hand_type=hand_type
    )
