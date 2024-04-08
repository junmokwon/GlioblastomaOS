import torch
from torch import nn
import torch.nn.functional as F
from utils import normalize_vector


class Prototype(nn.Module):
    def __init__(self, n_features, n_classes):
        super(Prototype, self).__init__()
        self.prototypes = nn.Parameter(torch.zeros((n_classes, n_features)), requires_grad=True)
        self.num_classes = n_classes
        nn.init.kaiming_normal_(torch.as_tensor(self.prototypes.data), mode="fan_out", nonlinearity="relu")
    
    def forward(self, x):
        return normalize_vector(x)
    
    def _get_pred_map(self, x):
        # Treat prototype weights as Conv3d weights
        p = self.prototypes
        weights = p.reshape(*list(p.size()) + [1, 1, 1])
        return F.conv3d(x, weights)
    
    def predict_survival(self, feature_maps, seg_preds):
        # Feature maps: (B, F, D, H, W)
        # Seg preds: (B, C, D, H, W)
        survival_preds_per_voxels = self._get_pred_map(feature_maps)
        survival_preds_per_voxels = normalize_vector(survival_preds_per_voxels)
        num_classes = self.num_classes
        masks = seg_preds

        # Voxel counts: (B, C)
        voxel_counts = masks.sum(dim=(2, 3, 4))

        # Calculate survival predictions only in foreground (aka masks) voxels
        preds = (survival_preds_per_voxels * masks).sum(dim=(2, 3, 4)) / voxel_counts.clamp(min=1)

        # Coeffs determines ratio of voxels among foreground classes
        coeffs = voxel_counts / voxel_counts.sum(dim=1).clamp(min=1)[:, None]
        return preds, coeffs

