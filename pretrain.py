from pathlib import Path
import os
import argparse
from contextlib import contextmanager
import torch
from monai.data import DataLoader, CacheDataset
from unetr import UNETRAE
from unetr_pp.unetr_pp_tumor import UNETR_PP_AE
from utils import *
from loss import dice_focal_loss


class UNETRAEBuilder(ModelBuilder):
    @staticmethod
    def get_input_size():
        return [96, 96, 96]
    
    @staticmethod
    def create_model():
        return UNETRAE(
            img_size=[96, 96, 96],
            in_channels=4,
            out_channels=3,
            feature_size=16,
            hidden_size=768,
            norm_name="instance",
            res_block=True,
            spatial_dims=3,
        )
    
    @staticmethod
    def get_feature_dim():
        return 16


class UNETRPPAEBuilder(ModelBuilder):
    @staticmethod
    def get_input_size():
        return [96, 96, 96]
    
    @staticmethod
    def create_model():
        return UNETR_PP_AE(
            in_channels=4,
            out_channels=3,
            feature_size=16,
            num_heads=4,
            depths=[3, 3, 3, 3],
            dims=[32, 64, 128, 256],
            do_ds=True,
            recon=True,
        )
    
    @staticmethod
    def get_feature_dim():
        return 16
    
    @staticmethod
    def deep_supervision():
        return True


def pretrain(args):
    model = args.model
    models = {
        'unetr': UNETRAEBuilder,
        'unetrpp': UNETRPPAEBuilder,
    }
    if model not in models.keys():
        raise RuntimeError(f'Invalid model "{model}"; Available models: {list(models.keys())}')
    opt = args.optimizer
    nnunet_config = args.nnunet_config
    if nnunet_config:
        opt = 'sgd'
    opts = {
        'adam': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
    }
    if opt not in opts.keys():
        raise RuntimeError(f'Invalid optimizer "{opt}"; Available optimizers: {list(opts.keys())}')
    model_builder = models[model]()
    pbar = ProgressBar()
    pbar.set_model(model)
    process_dir = Path(args.output_dir)
    if not process_dir.exists():
        if process_dir.parent.exists():
            os.umask(0)
            process_dir.mkdir(parents=True, exist_ok=False)
        else:
            raise FileNotFoundError(process_dir)
    pretrain_dir = process_dir / 'pretrain'
    if not pretrain_dir.exists():
        os.umask(0)
        pretrain_dir.mkdir(parents=True, exist_ok=False)
    dataset = load_brats20(Path(args.input_dir), type='LGG')
    train_transform, _ = get_transforms(model_builder.get_input_size())
    train_ds = CacheDataset(
        data=dataset,
        transform=train_transform,
        cache_rate=args.cache_rate
    )
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    cuda = args.cuda
    device = torch.device('cuda' if cuda else 'cpu')
    model = model_builder.create_model()
    model.to(device)
    initial_lr = args.initial_lr
    if initial_lr <= 0:
        default_initial_lr = {
            'adam': 1e-4,
            'sgd': 1e-2,
        }
        initial_lr = default_initial_lr[opt]
    opt_params = {}
    if opt == 'sgd':
        opt_params['momentum'] = 0.9
        opt_params['nesterov'] = True
        opt_params['weight_decay'] = 3e-5
    optimizer = opts[opt](model.parameters(), initial_lr, **opt_params)
    max_epochs = args.max_epochs
    if nnunet_config:
        lr_scheduler = PolyLR(optimizer, power=0.9, max_epoch=max_epochs)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    amp = args.amp
    if cuda and amp:
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast
    else:
        @contextmanager
        def autocast():
            yield
    
    loaders = [train_loader]
    num_batches = sum(map(len, loaders))
    pbar.set_total(max_epochs * num_batches)
    model.train()
    for epoch in range(1, max_epochs + 1):
        pbar.set_epoch('train', epoch)
        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels_one_hot = batch_data["label"].to(device)
            multi_channels = batch_data["multi_channel"].to(device)
            names = batch_data["name"]
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                seg_preds = outputs[0]
                if model.training and model_builder.deep_supervision():
                    n_ds_dim = seg_preds.size(1)
                    seg_loss = sum(map(lambda pred: dice_focal_loss(pred, multi_channels), torch.unbind(seg_preds, dim=1)))
                    seg_loss /= n_ds_dim
                    seg_preds = seg_preds[:, 0, ...]
                else:
                    seg_loss = dice_focal_loss(seg_preds, multi_channels)
                loss = seg_loss
            if cuda and amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            pbar.update(inputs.shape[0])
        lr_scheduler.step()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }, str(pretrain_dir / 'latest_model.pth.tar'))


def main():
    parser = argparse.ArgumentParser(description='Pretrain GlioblastomaOS')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--cache_rate', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--initial_lr', type=float, default=-1)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--model', type=str, default='unetrpp')
    parser.add_argument('--nnunet_config', action='store_true')
    parser.set_defaults(cuda=True, amp=True, nnunet_config=True)
    args = parser.parse_args()
    pretrain(args)


if __name__ == '__main__':
    main()
