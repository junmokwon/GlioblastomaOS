# GlioblastomaOS

Please navigate [`loss.py`](./loss.py) , [`prototype.py`](./prototype.py) , and [`train.py`](./train.py) for implementation details.

## Citation
If you find this code useful in your research, please consider citing:

```
@article{kwon2024leveraging,
    title={Leveraging segmentation-guided spatial feature embedding for overall survival prediction in glioblastoma with multimodal magnetic resonance imaging},
    author={Kwon, Junmo and Kim, Jonghun and Park, Hyunjin},
    journal={Computer Methods and Programs in Biomedicine},
    volume={255},
    pages={108338},
    year={2024},
    doi={10.1016/j.cmpb.2024.108338},
    publisher={Elsevier}
}
```

## Simple Usage

- Pretrain segmentation network only

```bash
python pretrain.py --model unetrpp --nnunet_config --input_dir /path/to/BraTS2020_training_data --output_dir /path/to/output --batch_size 8
```

- Train the whole model with pretrained segmentation network

```bash
python train.py --model unetrpp --nnunet_config --input_dir /path/to/BraTS2020_training_data --output_dir /path/to/output --batch_size 8
```
