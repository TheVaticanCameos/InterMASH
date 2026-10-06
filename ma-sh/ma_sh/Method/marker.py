import manifold_py
import trimesh
import os
import open3d as o3d
import numpy as np
import argparse


def mark_inside_faces(ply_path: str):
    original_mesh = trimesh.load(ply_path)
    manifold_mesh = get_manifold(ply_path)
    # Remove degenerate faces and simplify large meshes
    manifold_mesh.update_faces(manifold_mesh.nondegenerate_faces())
    manifold_mesh = manifold_mesh.simplify_quadric_decimation(face_count=2000, aggression=1)  # Adjust the face count as needed
    manifold_mesh.export("/data/yxz/MASH-HOI/MASH/test_template_manifold.ply")
    # Find triangle IDs in original_mesh whose three vertices are all inside manifold_mesh
    # Relax the threshold: treat distances <= 0.1 as inside; adjust tolerance as needed
    # tolerance = 3e-5
    tolerance = 1e-5
    signed_dist = manifold_mesh.nearest.signed_distance(original_mesh.vertices)
    # import pdb
    # pdb.set_trace()
    # Only treat negative distances (strictly inside) or distances at or below the negative threshold as inside, avoiding false positives for nearby points
    inside_mask = signed_dist <= -tolerance
    # Each face consists of three vertex indices; check whether all three vertices are inside
    inside_faces = []
    for face_id, face in enumerate(original_mesh.faces):
        if inside_mask[face].all():
            inside_faces.append(face_id)
    return inside_faces


def get_manifold(ply_path: str) -> trimesh.Trimesh:
    original_mesh = trimesh.load(ply_path)
    temp_obj_path = f"/home/yxz/temp/temp.obj"
    temp_manifold_path = f"/home/yxz/temp/temp_manifold.obj"
    original_mesh.export(temp_obj_path)
    manifold_py.run(temp_obj_path, temp_manifold_path)
    manifold_mesh = trimesh.load(temp_manifold_path)
    
    if os.path.exists(temp_obj_path):
        os.remove(temp_obj_path)
    if os.path.exists(temp_manifold_path):
        os.remove(temp_manifold_path)
    
    return manifold_mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化内部面或内外面分色")
    parser.add_argument("--mode", choices=["inside", "both"], default="inside",
                        help="inside: 仅显示内部面；both: 内外面分色显示")
    args = parser.parse_args()

    ply_path = "/data2/yxz/DATASET/DexGraspNet/Processed/shadow_template_hand.ply"
    inside_faces = mark_inside_faces(ply_path)
    print(len(inside_faces))

    # Load the original mesh
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()

    v_np = np.asarray(mesh.vertices)
    f_np = np.asarray(mesh.triangles)
    inside_set = set(inside_faces)

    new_vertices = []
    new_triangles = []
    new_colors = []

    color_in = np.array([1.0, 0.0, 0.0])   # Red
    color_out = np.array([0.0, 0.0, 1.0])  # Blue

    for i, tri in enumerate(f_np):
        if args.mode == "inside" and i not in inside_set:
            continue
        verts = v_np[tri]  # (3, 3)
        base = len(new_vertices)
        new_vertices.extend(verts.tolist())
        new_triangles.append([base, base + 1, base + 2])
        if i in inside_set:
            new_colors.extend([color_in.tolist(), color_in.tolist(), color_in.tolist()])
        else:
            new_colors.extend([color_out.tolist(), color_out.tolist(), color_out.tolist()])

    colored_mesh = o3d.geometry.TriangleMesh()
    colored_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(new_vertices))
    colored_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(new_triangles))
    colored_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(new_colors))
    colored_mesh.compute_vertex_normals()

    # Visualize
    if args.mode == "inside":
        o3d.visualization.draw_geometries([colored_mesh], window_name="Inside Faces Only (Red)")
    else:
        o3d.visualization.draw_geometries([colored_mesh], window_name="Inside (Red) vs Outside (Blue) Faces")
