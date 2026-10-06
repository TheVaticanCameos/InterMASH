import torch

from ma_sh.Module.hand_trainer import HandTrainer


class Convertor(object):
    def __init__(
        self,
        hand_path: str,
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
        template_info_path: str = None,
        hand_type: str = "mano"
    ) -> None:
        self.hand_path = hand_path
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
        self.template_info_path = template_info_path
        self.hand_type = hand_type

        self.error_dict = {}

        return

    def createTrainer(self) -> HandTrainer:
        trainer = HandTrainer(
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
            warmup_step_num=self.warmup_step_num,
            patience=self.patience,
            save_log_folder_path=self.save_log_folder,
            hand_template_info_path=self.template_info_path,
        )
        return trainer

    def convertData(self) -> bool:
        trainer = self.createTrainer()
        if not trainer.loadMeshFile(self.hand_path):
            return False
        if self.train_method == "escalate":
            if not trainer.escalateTrainMash():
                return False
        else:
            return False
        if not trainer.saveHandMash(self.output_path):
            return False
        self.error_dict = {
            "error": trainer.error
        }
        return True
