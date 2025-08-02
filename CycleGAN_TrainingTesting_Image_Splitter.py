#!/usr/bin/env python3
import os
import random
import shutil

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    hq_dir = os.path.join(base_dir, 'data', 'HQ_DWI')
    lq_base_dir = os.path.join(base_dir, 'data', 'LQ_DWI')

    dwi_dataset_dir = os.path.join(
        base_dir,
        'pytorch-CycleGAN-and-pix2pix',
        'datasets',
        'DWI'
    )
    trainB = os.path.join(dwi_dataset_dir, 'trainB')
    testB  = os.path.join(dwi_dataset_dir, 'testB')
    trainA = os.path.join(dwi_dataset_dir, 'trainA')
    testA  = os.path.join(dwi_dataset_dir, 'testA')

    for d in [trainA, trainB, testA, testB]:
        os.makedirs(d, exist_ok=True)

    # HQ split
    hq_files = [
        os.path.join(hq_dir, f)
        for f in os.listdir(hq_dir)
        if os.path.isfile(os.path.join(hq_dir, f)) and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    random.shuffle(hq_files)
    split_idx = int(0.7 * len(hq_files))
    hq_train = hq_files[:split_idx]
    hq_test  = hq_files[split_idx:]
    for f in hq_train:
        shutil.copy2(f, trainB)
    for f in hq_test:
        shutil.copy2(f, testB)

    # LQ split by folder
    lq_folders = [
        os.path.join(lq_base_dir, d)
        for d in os.listdir(lq_base_dir)
        if os.path.isdir(os.path.join(lq_base_dir, d))
    ]
    random.shuffle(lq_folders)
    split_idx = int(0.7 * len(lq_folders))
    lq_train_folders = lq_folders[:split_idx]
    lq_test_folders  = lq_folders[split_idx:]

    # Copy LQ images
    for folder in lq_train_folders:
        for f in os.listdir(folder):
            src = os.path.join(folder, f)
            if os.path.isfile(src) and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                shutil.copy2(src, trainA)
    for folder in lq_test_folders:
        for f in os.listdir(folder):
            src = os.path.join(folder, f)
            if os.path.isfile(src) and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                shutil.copy2(src, testA)

    print(f"HQ -> trainB: {len(hq_train)} images, testB: {len(hq_test)} images")
    lq_train_count = sum(
        len([f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
        for d in lq_train_folders
    )
    lq_test_count = sum(
        len([f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
        for d in lq_test_folders
    )
    print(f"LQ -> trainA: {lq_train_count} images, testA: {lq_test_count} images")

if __name__ == '__main__':
    main()
