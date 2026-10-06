import os
import sys
import functools

from ma_sh.Model.mash import Mash
from ma_sh.Model.hoi_mash import HOIMash
from ma_sh.Model.hand_mash import HandMash
from ma_sh.Model.object_mash import ObjectMash
from ma_sh.Module.o3d_viewer import O3DViewer


def compare(str_a: str, str_b: str) -> int:
    id_a = int(str_a.split("_")[0])
    id_b = int(str_b.split("_")[0])

    if id_a > id_b:
        return 1
    elif id_a == id_b:
        return 0
    return -1


def demo_view_mash(mash_file_path: str) -> bool:
    mash = Mash.fromParamsFile(mash_file_path, 20, 20, device="cuda:0")

    print("start show mash:", mash_file_path)
    mash.renderSamplePoints(show_positions=True)
    mash.renderPositions()
    return True


def demo_view_hand(
    mash_file_path: str,
    show_positions: bool = True,
    show_hand_line: bool = False,
) -> bool:
    hand_mash = HandMash.fromParamsFile(mash_file_path, 32, 32, device="cuda:0")

    print("start show hand mash:", mash_file_path)
    hand_mash.renderSamplePoints(show_positions=show_positions, show_hand_line=show_hand_line)
    return True


def demo_view_object(
    mash_file_path: str,
    show_positions: bool = True,
    show_object_line: bool = False,
) -> bool:
    object_mash = ObjectMash.fromParamsFile(mash_file_path, 32, 32, device="cuda:0")

    print("start show object mash:", mash_file_path)
    object_mash.renderSamplePoints(show_positions=show_positions, show_object_line=show_object_line)
    return True


def demo_view_hoi(
    mash_file_path: str, 
    show_mash: bool, 
    show_line_hand: bool,
    show_line_object: bool, 
    show_positions: bool, 
    target_name: str,
    feat_name: str,
    save_path: str = None,
    show_cmap: bool = False
) -> bool:
    hoi_mash = HOIMash.fromParamsFile(mash_file_path, 64, 16, device="cuda:0")
    
    print("start show HOI Mash:", mash_file_path)
    if show_mash:
        hoi_mash.renderSamplePoints(
            show_positions=True, 
            save_path=save_path, 
            show_hand_line=show_line_hand, 
            show_object_line=show_line_object,
            show_cmap=show_cmap
        )
    if show_positions:
        hoi_mash.renderPositions(feat_name, target_name)
    
    return True


def demo_view_folder(mash_folder_path: str) -> bool:
    if mash_folder_path[-1] != "/":
        mash_folder_path += "/"

    sample_phi_num = 40
    sample_theta_num = 40

    mash_filename_list = os.listdir(mash_folder_path)
    mash_filename_list.sort()

    for mash_filename in mash_filename_list:
        mash_file_path = mash_folder_path + mash_filename

        mash = Mash.fromParamsFile(
            mash_file_path,
            sample_phi_num,
            sample_theta_num,
        )

        print("start show mash:", mash_file_path)
        mash.renderSamplePoints()
    return True


def demo_view_training(mash_folder_path: str, view_freq: int = 1) -> bool:
    if mash_folder_path[-1] != "/":
        mash_folder_path += "/"

    sample_phi_num = 40
    sample_theta_num = 40

    o3d_viewer = O3DViewer()
    o3d_viewer.createWindow()

    print("start view:", mash_folder_path)

    mash_filename_list = os.listdir(mash_folder_path)
    mash_filename_list.sort(key=functools.cmp_to_key(compare))

    for i, mash_filename in enumerate(mash_filename_list):
        if i + 1 != len(mash_filename_list):
            if (i + 1) % view_freq != 0:
                continue

        mash_file_path = mash_folder_path + mash_filename

        mash = Mash.fromParamsFile(
            mash_file_path,
            sample_phi_num,
            sample_theta_num,
        )

        pcd = mash.toSamplePcd()

        o3d_viewer.clearGeometries()
        o3d_viewer.addGeometry(pcd)

        print("now render is", i + 1, "/", len(mash_filename_list))

        o3d_viewer.update()

    o3d_viewer.run()
    return True
