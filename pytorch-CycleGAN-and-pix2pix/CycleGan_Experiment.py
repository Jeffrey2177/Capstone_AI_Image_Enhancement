#!/usr/bin/env python3
import subprocess
import sys
import os

# ----------------------------------------
# Configuration - modify as needed
# ----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATAROOT = os.path.join(BASE_DIR, 'datasets', 'DWI')
EXPERIMENT_NAME = 'cycle_test'
MODEL = 'cycle_gan'
DIRECTION = 'AtoB'
# Image sizes
LOAD_SIZE = 512
CROP_SIZE = 512
# How many test images to process
NUM_TEST = 1
# GPU setting: '-1' for CPU only, or '0' for CUDA:0
GPU_IDS = '-1'
# ----------------------------------------

def make_test_cmd():
    return [
        sys.executable,
        os.path.join(BASE_DIR, 'test.py'),
        '--dataroot', DATAROOT,
        '--name', EXPERIMENT_NAME,
        '--model', MODEL,
        '--direction', DIRECTION,
        '--load_size', str(LOAD_SIZE),
        '--crop_size', str(CROP_SIZE),
        '--num_test', str(NUM_TEST),
        '--gpu_ids', GPU_IDS
    ]

if __name__ == '__main__':
    cmd = make_test_cmd()
    print('Running inference with command:')
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
