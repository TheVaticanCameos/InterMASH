# import os
# import sys

# sys.path.append("../chamfer-distance")
# sys.path.append("../data-convert")
# sys.path.append("../diff-curvature")
# sys.path.append("../mesh-graph-cut")
# sys.path.append("../mesh-cut")
# sys.path.append("../mesh-sample")
# sys.path.append("../sdf-generate")

# from ma_sh.Demo.pipeline_convertor import demo_convert_hoi

# if __name__ == "__main__":
    
#     hand_template_info_path = "/data2/yxz/DATASET/template_info/sphere_mano_template.npy"
#     object_template_info_path = "/data2/yxz/DATASET/GRAB-Original/object_templates/banana.npy"

#     hand_path = "/data2/yxz/DATASET/GRAB-Original/train/mano_banana_129_hand.ply"
#     object_path = "/data2/yxz/DATASET/GRAB-Original/train/mano_banana_129_object.ply"
    
#     output_path = "./ablation_hoi_direct.npy"
#     save_log_folder = "/data/yxz/MASH-HOI/MASH/ma-sh/ablation/log/direct"
#     cuda_id = "0"

#     os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    
#     demo_convert_hoi(
#         hand_path=hand_path, 
#         object_path=object_path, 
#         output_path=output_path, 
#         save_log_folder=save_log_folder, 
#         anchor_num=256, 
#         hand_template_info_path=hand_template_info_path,
#         object_template_info_path=object_template_info_path,
#         hand_type="mano"
#     )

from ma_sh.Model.hoi_mash import HOIMash
import trimesh


direct_file_path = "/data/yxz/MASH-HOI/MASH/ma-sh/ablation_hoi_direct.npy"
staged_file_path = "/data/yxz/MASH-HOI/MASH/ma-sh/ablation_hoi_staged.npy"

direct_mash = HOIMash.fromParamsFile(direct_file_path, 64, 16, device="cuda:0")
staged_mash = HOIMash.fromParamsFile(staged_file_path, 64, 16, device="cuda:0")

direct_hand_pts = direct_mash.toSamplePoints("hand").reshape(-1, 3).cpu().numpy()
staged_hand_pts = staged_mash.toSamplePoints("hand").reshape(-1, 3).cpu().numpy()

direct_obj_pts = direct_mash.toSamplePoints("object").reshape(-1, 3).cpu().numpy()
staged_obj_pts = staged_mash.toSamplePoints("object").reshape(-1, 3).cpu().numpy()

trimesh.PointCloud(direct_hand_pts).export("direct_hand_pts.ply")
trimesh.PointCloud(staged_hand_pts).export("staged_hand_pts.ply")

trimesh.PointCloud(direct_obj_pts).export("direct_obj_pts.ply")
trimesh.PointCloud(staged_obj_pts).export("staged_obj_pts.ply")
