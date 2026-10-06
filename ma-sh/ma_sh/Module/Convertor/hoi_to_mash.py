import torch
from time import time

from ma_sh.Module.hoi_trainer import HOITrainer
from ma_sh.Module.hoi_trainer_seperate import HOITrainerSeperate
# from data_convert.Module.base_convertor import BaseConvertor


class Convertor(object):
    def __init__(
        self,
        hand_path: str, 
        object_path: str, 
        output_path: str, 
        initial_method: str = "voxel",
        train_method: str = "auto",
        anchor_num: int = 4096,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 40,
        sample_theta_num: int = 40,
        points_per_submesh: int = 1024,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        lr: float = 2e-3,
        min_lr: float = 1e-3,
        warmup_step_num: int = 80,
        warmup_epoch: int = 4,
        factor: float = 0.8,
        patience: int = 2,
        direct_epoch_num: int = 800,
        render: bool = False,
        save_log_folder: str = None,
        hand_template_info_path: str = None,
        object_template_info_path: str | None = None,
        hand_type: str = "mano",
        cmap_path: str = None,
        use_closed_form_cmap: bool = True
    ) -> None:
        self.hand_path = hand_path
        self.object_path = object_path
        self.output_path = output_path

        self.initial_method = initial_method
        self.anchor_num = anchor_num
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.sample_phi_num = sample_phi_num
        self.sample_theta_num = sample_theta_num
        self.points_per_submesh = points_per_submesh
        self.dtype = dtype
        self.device = device

        self.train_method = train_method
        self.direct_epoch_num = direct_epoch_num
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_step_num = warmup_step_num
        self.warmup_epoch = warmup_epoch
        self.factor = factor
        self.patience = patience
        self.render = render
        self.save_log_folder = save_log_folder
        self.hand_template_info_path = hand_template_info_path
        self.object_template_info_path = object_template_info_path

        self.hand_type = hand_type
        self.cmap_path = cmap_path

        self.error_dict = {}

        self.use_closed_form_cmap = use_closed_form_cmap

        return

    # def createTrainer(self) -> HOITrainerSeperate:
    #     trainer = HOITrainerSeperate(
    #         initial_method=self.initial_method,
    #         anchor_num=self.anchor_num,
    #         mask_degree_max=self.mask_degree_max,
    #         sh_degree_max=self.sh_degree_max,
    #         sample_phi_num=self.sample_phi_num,
    #         sample_theta_num=self.sample_theta_num,
    #         points_per_submesh=self.points_per_submesh,
    #         dtype=self.dtype,
    #         device=self.device,
    #         lr=self.lr,
    #         min_lr=self.min_lr,
    #         warmup_step_num=self.warmup_step_num,
    #         warmup_epoch=self.warmup_epoch,
    #         factor=self.factor,
    #         patience=self.patience,
    #         render=self.render,
    #         save_log_folder_path=self.save_log_folder,
    #         template_info_path=self.template_info_path,
    #         hand_type=self.hand_type,
    #     )

    #     return trainer

    def createTrainer(self) -> HOITrainer:
        trainer = HOITrainer(
            initial_method=self.initial_method,
            anchor_num=self.anchor_num,
            mask_degree_max=self.mask_degree_max,
            sh_degree_max=self.sh_degree_max,
            sample_phi_num=self.sample_phi_num,
            sample_theta_num=self.sample_theta_num,
            points_per_submesh=self.points_per_submesh,
            dtype=self.dtype,
            device=self.device,
            lr=self.lr,
            min_lr=self.min_lr,
            warmup_step_num=self.warmup_step_num,
            warmup_epoch=self.warmup_epoch,
            factor=self.factor,
            patience=self.patience,
            render=self.render,
            save_log_folder_path=self.save_log_folder,
            hand_template_info_path=self.hand_template_info_path,
            object_template_info_path=self.object_template_info_path,
            hand_type=self.hand_type,
            use_closed_form_cmap=self.use_closed_form_cmap
        )

        return trainer

    def convertData(self) -> bool:

        trainer = self.createTrainer()

        if not trainer.loadMeshFile(self.hand_path, self.object_path, self.cmap_path):
            print("[ERROR][Convertor::convertData]")
            print("\t loadMeshFile failed!")
            return False

        if self.train_method == "auto":
            if not trainer.autoTrainMash():
                print("[ERROR][Convertor::convertData]")
                print("\t autoTrainMash failed!")
                return False
        elif self.train_method == "direct":
            if not trainer.directTrainMash(self.direct_epoch_num):
                print("[ERROR][Convertor::convertData]")
                print("\t directTrainMash failed!")
                return False
        elif self.train_method == "escalate":
            if not trainer.escalateTrainMash():
                print("[ERROR][Convertor::convertData]")
                print("\t escalateTrainMash failed!")
                return False
        else:
            print("[ERROR][Convertor::convertData]")
            print("\t train_method not support!")
            return False

        if not trainer.saveMashFile(self.output_path):
            print("[ERROR][Convertor::convertData]")
            print("\t saveMashFile failed!")
            return False
        
        self.error_dict = {
            "error": trainer.error,
            "hand_error": trainer.hand_error,
            "object_error": trainer.object_error
        }

        return True
