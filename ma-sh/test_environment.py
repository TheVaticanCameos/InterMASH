import os
import sys

sys.path.append("../chamfer-distance")
sys.path.append("../data-convert")
sys.path.append("../diff-curvature")
sys.path.append("../mesh-graph-cut")
sys.path.append("../mesh-cut")
sys.path.append("../mesh-sample")
sys.path.append("../sdf-generate")

from ma_sh.Demo.pipeline_convertor import demo_convert_hoi

if __name__ == "__main__":
    name = "airplane_fly_1_0"

    hand_path = "./data/" + name + "_hand.ply"
    object_path = "./data/" + name + "_object.ply"
    
    output_path = "./data/" + name + "_hoi.npy"
    
    save_log_folder = "./data/log"
    cuda_id = "0"

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    
    demo_convert_hoi(
        hand_path, 
        object_path, 
        output_path, 
        save_log_folder, 
        anchor_num=256, 
        template_path="./data/hand_template.npy", 
        hand_type="mano"
    )
    