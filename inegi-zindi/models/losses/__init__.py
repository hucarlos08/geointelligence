from .center_loss import(
    CenterLoss
)


from .arcface_loss import(
    ArcFaceLoss
)

from .focal_loss import(
    FocalLoss
)

from .sigmoid_focal_loss import(
    SigmoidFocalLoss
)

from .combined_loss import(
    CombinedLoss
)

__all__ = [
    'CenterLoss'
    'ArcFaceLoss',
    'FocalLoss',
    'SigmoidFocalLoss'
]