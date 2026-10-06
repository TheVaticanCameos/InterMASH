from .LossModule.loss_collision import CollisionLossModule
from .LossModule.loss_collision_multihand import MultiHandCollisionLossModule
# from .LossModule.loss_collision_multi_only_hand import MultiHandCollisionLossModuleOnlyHand


__all__ = [
    "CollisionLossModule",
    "MultiHandCollisionLossModule",
    # "MultiHandCollisionLossModuleOnlyHand"
]
