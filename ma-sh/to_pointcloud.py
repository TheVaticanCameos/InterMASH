from ma_sh.Model.hoi_mash import HOIMash
import trimesh


file_path = "/data/yxz/MASH-HOI/MASH/mash_hoi/Results/MASH/exp_ddpm_dga_loss_dro_handcmap/gen_shadow_ycb+baseball_object_0.npy"

hoi_mash = HOIMash.fromParamsFile(file_path, 64, 16, device="cuda:0")
hand_pts = hoi_mash.toSamplePoints("hand").reshape(-1, 3).cpu().numpy()

trimesh.PointCloud(hand_pts).export("gt_pts_max.ply")

