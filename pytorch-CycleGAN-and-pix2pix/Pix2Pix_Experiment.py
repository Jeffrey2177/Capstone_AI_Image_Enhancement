# Pix2Pix_experiment.py

import os
import sys
import subprocess
import shutil
from pathlib import Path

ROOT        = Path(__file__).resolve().parent
TEST_SCRIPT = ROOT / "test.py"


def run_on_test(
    dataroot: str,
    name: str,
    direction: str = "AtoB",
    load_size: int = 512,
    crop_size: int = 512,
    gpu_ids: int = -1,
    num_test: int | None = None
):
    """
    Invokes: python test.py --dataroot <dataroot> --name <name> --model pix2pix ...
    Loads checkpoints/<name>/latest_net_G.pth and writes outputs to:
      results/<name>/test_latest/images/
    """
    cmd = [
        sys.executable, str(TEST_SCRIPT),
        "--dataroot",  dataroot,
        "--name",      name,
        "--model",     "pix2pix",
        "--phase",     "test",
        "--direction", direction,
        "--load_size", str(load_size),
        "--crop_size", str(crop_size),
        "--gpu_ids",   str(gpu_ids),
        "--batch_size","1",
    ]
    if num_test is not None:
        cmd += ["--num_test", str(num_test)]

    print("\nRunning test with command:\n", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)


def option_1_my_latest():
    """
    Test using your own latest-trained generator in:
      checkpoints/pix2pix_main/latest_net_G.pth
    Results → results/pix2pix_main/test_latest/images
    """
    dataroot = "data"
    # count how many AB‐pair PNGs are in data/test/
    test_dir = Path(dataroot) / "test"
    num = len(list(test_dir.glob("*.png")))

    run_on_test(
        dataroot=dataroot,
        name="pix2pix_main",
        direction="AtoB",
        num_test=num
    )


def option_2_pretrained():
    """
    Test using one of two downloaded pretrained G models.

    [1] Facades:      checkpoints/pretrained/facade_pretrained/facades_label2photo.pth
    [2] Edges2Shoes:  checkpoints/pretrained/edges2shoes_pretrained/edges2shoes.pth

    The chosen model is copied to checkpoints/<model_key>/latest_net_G.pth
    and run with --name <model_key> → results/<model_key>/test_latest/images
    """
    print("\nChoose pretrained model to test:")
    print("  [1] facades_label2photo")
    print("  [2] edges2shoes")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        key = "facades_label2photo"
        src = ROOT / "checkpoints" / "pretrained" / "facade_pretrained" / "facades_label2photo.pth"
    elif choice == "2":
        key = "edges2shoes"
        src = ROOT / "checkpoints" / "pretrained" / "edges2shoes_pretrained" / "edges2shoes.pth"
    else:
        print("Invalid choice. Exiting.")
        return

    dst_dir = ROOT / "checkpoints" / key
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_pth = dst_dir / "latest_net_G.pth"
    if not dst_pth.exists():
        print(f"Copying pretrained G → {dst_pth}")
        shutil.copy(src, dst_pth)

    dataroot = "data"
    # count how many AB‐pair PNGs are in data/test/
    test_dir = Path(dataroot) / "test"
    num = len(list(test_dir.glob("*.png")))

    run_on_test(
        dataroot=dataroot,
        name=key,
        direction="AtoB",
        num_test=num
    )


if __name__ == "__main__":
    print("""
Choose which generator to test:
  [1] My latest-trained model (pix2pix_main)
  [2] Downloaded pretrained models
""")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        option_1_my_latest()
    elif choice == "2":
        option_2_pretrained()
    else:
        print("Invalid choice. Exiting.")
