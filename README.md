# GlioblastomaOS

Please navigate `loss.py` , `prototype.py` , and `train.py`  for implementation details.

# Simple Usage

- Pretrain segmentation network only

```bash
python pretrain.py --input_dir /path/to/MICCAI_BraTS17_Data_Training --output_dir /path/to/output --batch_size 8
```

- Train the whole model with pretrained segmentation network

```bash
python train.py /path/to/MICCAI_BraTS17_Data_Training --output_dir /path/to/output --batch_size 8
```