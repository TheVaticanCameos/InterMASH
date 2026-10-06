import os
import sys

sys.path.append("../chamfer-distance")
sys.path.append("../data-convert")
sys.path.append("../diff-curvature")
sys.path.append("../mesh-graph-cut")
sys.path.append("../mesh-cut")
sys.path.append("../mesh-sample")
sys.path.append("../sdf-generate")

from ma_sh.Demo.pipeline_convertor import demo_convert_hoi_cmap

if __name__ == "__main__":
    name = "ycb+gelatin_box_97"

    template_info_path = "/data2/yxz/DATASET/template_info/barrett_template.npy"
    # template_info_path = "/data2/yxz/DATASET/DexGraspNet/Processed/hand_template.npy"
    hand_path = "/data2/yxz/DATASET/DRO-data/DRO-Original/barrett/train/barrett_" + name + "_hand.ply"
    # hand_path = "/data2/yxz/DATASET/DexGraspNet/Processed/hand_core-bottle-1a7ba1f4c892e2da30711cdbdbc73924_0.ply"
    # hand_path = "/data2/yxz/DATASET/GrabDataset/hand_rest.ply"
    
    object_path = "/data2/yxz/DATASET/DRO-data/DRO-Original/barrett/train/barrett_" + name + "_object.ply"
    # object_path = "/data2/yxz/DATASET/DexGraspNet/Processed/object_core-bottle-1a7ba1f4c892e2da30711cdbdbc73924_0.ply"
    # output_path = "/data/yxz/MASH-HOI/MASH/ma-sh/test/test_hoi_" + name + ".npy"

    cmap_path = "/data2/yxz/DATASET/DRO-data/DRO-Original/barrett/train/barrett_" + name + "_cmap.npy"
    
    output_path = "./test_hoi.npy"
    save_log_folder = "/data/yxz/MASH-HOI/MASH/ma-sh/test/log"
    cuda_id = "1"

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    
    demo_convert_hoi_cmap(
        hand_path=hand_path, 
        object_path=object_path, 
        cmap_path=cmap_path,
        output_path=output_path, 
        save_log_folder=save_log_folder, 
        anchor_num=256, 
        template_info_path=template_info_path,
        hand_type="barrett",
        use_closed_form_cmap=True
    )
    
