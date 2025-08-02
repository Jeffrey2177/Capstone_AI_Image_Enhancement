#!/usr/bin/env python3
import subprocess
import sys
import os

# ----------------------------------------
# Configuration - modify as needed
# ----------------------------------------
# Root of your CycleGAN repo (this script's directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the datasets/DWI folder
DATAROOT = os.path.join(BASE_DIR, 'datasets', 'DWI')
# Experiment name (folder under checkpoints/ and results/)
EXPERIMENT_NAME = 'cycle_test'
# Model type
MODEL = 'cycle_gan'
# Direction of translation (A = LQ → B = HQ)
DIRECTION = 'AtoB'
# Batch and image sizes
BATCH_SIZE = 1
LOAD_SIZE = 512
CROP_SIZE = 512
# Training schedule
EPOCHS = 1
EPOCHS_DECAY = 0
SAVE_EPOCH_FREQ = 1
# GPU setting: '-1' for CPU only, or '0' (string) for CUDA:0
GPU_IDS = '-1'
# ----------------------------------------

def make_train_cmd():
    return [
        sys.executable,
        os.path.join(BASE_DIR, 'train.py'),
        '--dataroot', DATAROOT,
        '--name', EXPERIMENT_NAME,
        '--model', MODEL,
        '--direction', DIRECTION,
        '--batch_size', str(BATCH_SIZE),
        '--load_size', str(LOAD_SIZE),
        '--crop_size', str(CROP_SIZE),
        '--n_epochs', str(EPOCHS),
        '--n_epochs_decay', str(EPOCHS_DECAY),
        '--save_epoch_freq', str(SAVE_EPOCH_FREQ),
        '--gpu_ids', GPU_IDS
    ]

if __name__ == '__main__':
    cmd = make_train_cmd()
    print('Running training with command:')
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
