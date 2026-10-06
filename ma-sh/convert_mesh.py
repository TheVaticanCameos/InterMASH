import os
import sys

sys.path.append("../chamfer-distance")
sys.path.append("../data-convert")
sys.path.append("../diff-curvature")
sys.path.append("../mesh-graph-cut")
sys.path.append("../sdf-generate")

from ma_sh.Demo.pipeline_convertor import demo_convert_mesh


if __name__ == "__main__":
    """
    data_space: the root folder of the shape dataset
    output_space: the root folder of the generated results
    rel_data_path: data_file_path = data_space + rel_data_path
    """
    data_space = "/data2/yxz/DATASET/Original-Meshes/DexGraspNet/train/"
    output_space = "/data2/yxz/DATASET/Original-Meshes/DexGraspNet/mash/"
    rel_data_path = "shadow_core-bottle-1a7ba1f4c892e2da30711cdbdbc73924_0_hand.ply"
    cuda_id = "1"

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    demo_convert_mesh(data_space, output_space, rel_data_path)
