import logging
from pathlib import Path
import os
import json
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F

from monai.transforms import (
    LoadImaged,
    EnsureTyped,
    Orientationd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    MapTransform,
    Compose,
    CropForegroundd,
)
from monai.utils import ensure_tuple


logger = logging.getLogger(__name__)


def load_json(filename):
    with open(filename, "r") as fp:
        return json.load(fp)

def save_json(_dict, filename):
    with open(filename, "w") as fp:
        json.dump(_dict, fp, indent=4, sort_keys=True)


try:
    config = load_json(Path(__file__).parent / 'config.json')
except:
    logger.exception('Invalid configuration file found. Please download config.json in Github repo.')


class ModelBuilder:
    @staticmethod
    def get_input_size():
        raise NotImplementedError

    @staticmethod
    def create_model():
        raise NotImplementedError
    
    @staticmethod
    def get_feature_dim():
        raise NotImplementedError
    
    @staticmethod
    def deep_supervision():
        return False


class ProgressBar:
    def __init__(self):
        self.pbar = None
        self.model = ''
        self.prefix = ''
        self.epoch = 0
        self.step = 0
    
    def set_total(self, total: int):
        self.pbar = tqdm(total=total, unit='batch')
    
    def set_step(self, step: int):
        self.step = step
    
    def set_description(self):
        self.pbar.set_description(f"[{self.model}] {self.prefix} epoch {self.epoch}")
    
    def set_model(self, model: str):
        self.model = model
    
    def set_epoch(self, prefix: str, epoch: int):
        self.prefix = prefix
        self.epoch = epoch
        self.set_description()
    
    def set_postfix(self, **kwargs):
        self.pbar.set_postfix(**kwargs)
    
    def update(self, inc: int):
        self.pbar.update(inc)
    
    def close(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, power=0.9, max_epoch=1000, last_epoch=-1):
        self.power = power
        self.max_epoch = max_epoch
        self.last_epoch = last_epoch
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            lr = base_lr * (1.0 - (self.last_epoch / self.max_epoch)) ** self.power
            lrs.append(lr)
        return lrs


def normalize_vector(x: torch.Tensor):
    return F.normalize(x, p=2.0, dim=1)

def list_directories(path):
    return next(os.walk(path))[1]

def load_brats17(data_dir: Path, type: str):
    available_types = ('HGG', 'LGG')
    if type not in available_types:
        raise RuntimeError(f'Invalid type "{type}"; Available types: {available_types}')
    top_dir = data_dir
    if not top_dir.exists():
        raise FileNotFoundError(top_dir)
    has_survival_data = type == 'HGG'
    if has_survival_data:
        survival_path = top_dir / 'survival_data.csv'
        survival_data = pd.read_csv(str(survival_path), index_col=0, encoding='utf-8', engine='python').astype(float)
        min_survival_in_days = float(config['min_survival_days'])
        max_survival_in_days = float(config['max_survival_days'])
        survival_data = survival_data[ (survival_data.Survival >= min_survival_in_days) & (survival_data.Survival <= max_survival_in_days) ]
    data_dir = top_dir / type
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    logger.info(f'Loading BraTS17: {data_dir}')
    subjects = []
    input_modalities = config['input_modalities']
    for subj_name in list_directories(data_dir):
        subj_dir = data_dir / subj_name
        if has_survival_data and subj_name not in survival_data.index:
            logger.debug(f'Skip {subj_name}: missing survival')
            continue
        error = False
        for mod in ('flair', 'seg', 't1', 't1ce', 't2'):
            mod_file = subj_dir / f'{subj_name}_{mod}.nii.gz'
            if not mod_file.exists():
                error = True
                logger.debug(f'Skip {subj_name}: missing {mod}')
                break
        if error:
            continue
        row = {
            "image": [subj_dir / f"{subj_name}_{mod}.nii.gz" for mod in input_modalities],
            "name": subj_name,
            "label": subj_dir / f"{subj_name}_seg.nii.gz"
        }
        if has_survival_data:
            for col_name in ('Age', 'Survival'):
                value = survival_data.loc[subj_name, col_name]
                row[col_name.lower()] = value
        subjects.append(row)
    logger.info(f'{len(subjects)} subjects loaded')
    return subjects


def load_brats20(data_dir: Path, type: str):
    available_types = ('HGG', 'LGG')
    if type not in available_types:
        raise RuntimeError(f'Invalid type "{type}"; Available types: {available_types}')
    top_dir = data_dir
    if not top_dir.exists():
        raise FileNotFoundError(top_dir)
    name_mapping_path = top_dir / 'name_mapping.csv'
    name_mapping = pd.read_csv(str(name_mapping_path), index_col=-1, encoding='utf-8', engine='python', usecols=['Grade', 'BraTS_2020_subject_ID'])
    has_survival_data = type == 'HGG'
    if has_survival_data:
        survival_path = top_dir / 'survival_info.csv'
        survival_data = pd.read_csv(str(survival_path), index_col=0, encoding='utf-8', engine='python', usecols=['Brats20ID', 'Age', 'Survival_days']).astype(float)
        survival_data.rename(columns={"Survival_days": "Survival"}, inplace=True)
        min_survival_in_days = float(config['min_survival_days'])
        max_survival_in_days = float(config['max_survival_days'])
        survival_data = survival_data[ (survival_data.Survival >= min_survival_in_days) & (survival_data.Survival <= max_survival_in_days) ]
    data_dir = top_dir / 'BraTS20_Training'
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    logger.info(f'Loading {type} BraTS20: {data_dir}')
    subjects = []
    input_modalities = config['input_modalities']
    for subj_name in list_directories(data_dir):
        subj_dir = data_dir / subj_name
        if name_mapping.loc[subj_name, 'Grade'] != type:
            continue
        if has_survival_data and subj_name not in survival_data.index:
            logger.debug(f'Skip {subj_name}: missing survival')
            continue
        error = False
        for mod in ('flair', 'seg', 't1', 't1ce', 't2'):
            mod_file = subj_dir / f'{subj_name}_{mod}.nii'
            if not mod_file.exists():
                error = True
                logger.debug(f'Skip {subj_name}: missing {mod}')
                break
        if error:
            continue
        row = {
            "image": [subj_dir / f"{subj_name}_{mod}.nii" for mod in input_modalities],
            "name": subj_name,
            "label": subj_dir / f"{subj_name}_seg.nii"
        }
        if has_survival_data:
            for col_name in ('Age', 'Survival'):
                value = survival_data.loc[subj_name, col_name]
                row[col_name.lower()] = value
        subjects.append(row)
    logger.info(f'{len(subjects)} subjects loaded')
    return subjects


class ConvertBratsLabelToMultiChannel(MapTransform):
    def __init__(self, keys, target_keys=None):
        super().__init__(keys, allow_missing_keys=False)
        if target_keys is None:
            target_keys = self.keys
        self.target_keys = ensure_tuple(target_keys)
    
    def __call__(self, data):
        d = dict(data)
        for src_key, dst_key in self.key_iterator(d, self.target_keys):
            img = d[src_key]
            
            # if img has channel dim, squeeze it
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            
            # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
            # label 4 is ET
            result = [(img == 1) | (img == 4), (img == 1) | (img == 4) | (img == 2), img == 4]
            d[dst_key] = torch.stack(result, dim=0)
        return d


class BratsLabelToOneHot(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for src_key in self.key_iterator(d):
            img = d[src_key]
            
            # if img has channel dim, squeeze it
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            
            result = [img == 1, img == 2, img == 4]
            d[src_key] = torch.stack(result, dim=0)
        return d


def get_transforms(crop_shape):
    image_keys = ["image", "label"]
    image_keys_with_multi_channel = ["image", "label", "multi_channel"]
    
    train_transform = Compose(
        [
            LoadImaged(keys=image_keys, ensure_channel_first=True),
            EnsureTyped(keys=image_keys),
            Orientationd(keys=image_keys, axcodes="RAS"),
            CropForegroundd(keys=image_keys, source_key="label", k_divisible=crop_shape),
            RandSpatialCropd(keys=image_keys, roi_size=crop_shape, random_size=False),
            ConvertBratsLabelToMultiChannel(keys="label", target_keys="multi_channel"),
            BratsLabelToOneHot(keys="label"),
            RandFlipd(keys=image_keys_with_multi_channel, prob=0.5, spatial_axis=0),
            RandFlipd(keys=image_keys_with_multi_channel, prob=0.5, spatial_axis=1),
            RandFlipd(keys=image_keys_with_multi_channel, prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=image_keys, ensure_channel_first=True),
            EnsureTyped(keys=image_keys),
            Orientationd(keys=image_keys, axcodes="RAS"),
            ConvertBratsLabelToMultiChannel(keys="label", target_keys="multi_channel"),
            BratsLabelToOneHot(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    return train_transform, val_transform

def norm_survival(days):
    min_survival = float(config['min_survival_days'])
    max_survival = float(config['max_survival_days'])
    return (days - min_survival) / (max_survival - min_survival)

def denorm_survival(values):
    min_survival = float(config['min_survival_days'])
    max_survival = float(config['max_survival_days'])
    return values * (max_survival - min_survival) + min_survival

def brats_multi_channel_to_one_hot(d):
    tc_index = 0
    wt_index = 1
    et_index = 2
    
    # Assign tumor core, whole tumor, and enhancing tumor
    tc = d[:, tc_index, ...]
    wt = d[:, wt_index, ...]
    et = d[:, et_index, ...]

    # Get NEC/NET
    nec = tc.logical_xor(et)

    # Get edema
    ed = wt.logical_xor(tc)

    result = [nec, ed, et]
    return torch.stack(result, dim=1)
