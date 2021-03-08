import torch, torchvision

import os.path as osp
import os
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model
from mmcv.runner import set_random_seed
import mmcv


from mmcv import Config
cfg = Config.fromfile('./configs/recognition/tpn/tpn_video_kinetics_tiny_rgb.py')


# Modify dataset type and path
dataset_type = 'VideoDataset'

ROOT_DATA = './tools/data/kinetics400_tiny'
ann_file_train = os.path.join(ROOT_DATA, 'kinetics_tiny_train_video.txt')


ann_file_val = os.path.join(ROOT_DATA, 'kinetics_tiny_val_video.txt')
ann_file_test = os.path.join(ROOT_DATA, 'kinetics_tiny_val_video.txt')
data_root_val = os.path.join(ROOT_DATA, 'val')
data_root = os.path.join(ROOT_DATA, 'train')

cfg.dataset_type = 'VideoDataset'
cfg.data_root = data_root
cfg.data_root_val =data_root_val
cfg.ann_file_train = ann_file_train
cfg.ann_file_val = ann_file_val
cfg.ann_file_test = ann_file_test

cfg.ann_file_train = os.path.join(ROOT_DATA,'kinetics_tiny_train_video.txt')
cfg.ann_file_val = os.path.join(ROOT_DATA,'kinetics_tiny_val_video.txt')
cfg.ann_file_test = os.path.join(ROOT_DATA,'kinetics_tiny_val_video.txt')

cfg.data.test.type = 'VideoDataset'
cfg.data.test.ann_file = os.path.join(ROOT_DATA,'kinetics_tiny_val_video.txt')
cfg.data.test.data_prefix = os.path.join(ROOT_DATA, 'val')

cfg.data.train.type = 'VideoDataset'
cfg.data.train.ann_file = os.path.join(ROOT_DATA,'kinetics_tiny_train_video.txt')
cfg.data.train.data_prefix = os.path.join(ROOT_DATA, 'train')

cfg.data.val.type = 'VideoDataset'
cfg.data.val.ann_file = os.path.join(ROOT_DATA,'kinetics_tiny_val_video.txt')
cfg.data.val.data_prefix = os.path.join(ROOT_DATA, 'val')

# The flag is used to determine whether it is omnisource training
cfg.setdefault('omnisource', False)
# Modify num classes of the model in cls_head
cfg.model.cls_head.num_classes = 2
# We can use the pre-trained TSN model
# cfg.load_from = './checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'


# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.

cfg.optimizer.lr = cfg.optimizer.lr / 8 / 16
cfg.total_epochs = 15

# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 10
# We can set the log print interval to reduce the the times of printing log
cfg.log_config.interval = 5

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)


# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the recognizer
model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_model(model, datasets, cfg, distributed=False, validate=True)
