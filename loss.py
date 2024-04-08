import itertools
import torch
from monai.losses import DiceFocalLoss
from utils import normalize_vector


mae_loss = torch.nn.L1Loss(reduction='mean')
mse_loss = torch.nn.MSELoss(reduction='mean')
dice_focal_loss = DiceFocalLoss(sigmoid=True, reduction='mean', lambda_dice=0.5, lambda_focal=0.5)
cosine_embedding_loss = torch.nn.CosineEmbeddingLoss(margin=0.5, reduction='none')


def survival_loss(loss_fn, preds, coeffs, targets):
    loss = None
    # Coeffs: (B, C)
    # Preds: (B, C)
    # Loop over C dimension
    for preds, coeff in zip(torch.unbind(preds, 1), torch.unbind(coeffs, 1)):
        _loss = loss_fn(preds.unsqueeze(dim=1), targets) * coeff
        if len(_loss.shape) > 0:
            _loss = _loss.mean()
        if loss is None:
            loss = _loss
        else:
            loss += _loss
    return loss / coeffs.size(0)

def contrastive_loss(feature_maps, seg_preds, n_classes=3, eps=1e-4):
    # Feature maps: (B, F, D, H, W)
    # Seg preds: (B, C, D, H, W)
    feature_maps = normalize_vector(feature_maps)
    B = feature_maps.shape[0]
    F = feature_maps.shape[1]
    C = seg_preds.shape[1]
    
    # Get imaging features for each class
    seg_preds_sum = seg_preds.sum(dim=(0, 2, 3, 4)).clamp(min=eps)
    imaging_features = (
        feature_maps.unsqueeze(dim=1) * seg_preds.unsqueeze(dim=2)
    ).sum(dim=(0, 3, 4, 5)) / seg_preds_sum.unsqueeze(dim=1).expand(C, F)
    
    # Get combinations of inter-class pairs
    combinations = list(itertools.combinations(imaging_features, r=2))
    combinations = list(map(lambda x: torch.stack(list(x), dim=0), combinations))
    combinations = torch.stack(combinations, dim=0)
    input1 = combinations[:, 0, :]
    input2 = combinations[:, 1, :]
    
    negative_flag = torch.tensor([-1]).to(feature_maps.device).expand(input1.size(0))
    
    # Maximize cosine distance between inter-class pairs
    loss = cosine_embedding_loss(input1, input2, negative_flag).mean()
    return loss
