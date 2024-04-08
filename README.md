# GlioblastomaOS

Please navigate `loss.py` , `prototype.py` , and `train.py`  for implementation details.

# Simple Usage

- Pretrain segmentation network only

```bash
python pretrain.py --model unetrpp --nnunet_config --input_dir /path/to/BraTS2020_training_data --output_dir /path/to/output --batch_size 8
```

- Train the whole model with pretrained segmentation network

```bash
python train.py --model unetrpp --nnunet_config --input_dir /path/to/BraTS2020_training_data --output_dir /path/to/output --batch_size 8
```