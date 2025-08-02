import sys
import subprocess
import os
import re

# Path to the train.py script (assumes this file lives in the same folder)
TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train.py")
RESUME_EPOCHS = 5  # always train 5 more epochs on resume

def train_fresh(
    dataroot: str,
    name: str = "pix2pix_main",
    model: str = "pix2pix",
    direction: str = "AtoB",
    n_epochs: int = 10,
    n_epochs_decay: int = 3,
    batch_size: int = 1,
    gpu_ids: int = -1,
    load_size: int = 512,
    crop_size: int = 512,
    save_latest_freq: int = 1,
):
    """
    Start a fresh training run using images in <dataroot>/train.
    Saves both latest checkpoint every iter and per-epoch checkpoint.
    """
    train_dir = os.path.join(dataroot, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--dataroot",        dataroot,
        "--name",            name,
        "--model",           model,
        "--direction",       direction,
        "--n_epochs",        str(n_epochs),
        "--n_epochs_decay",  str(n_epochs_decay),
        "--batch_size",      str(batch_size),
        "--gpu_ids",         str(gpu_ids),
        "--load_size",       str(load_size),
        "--crop_size",       str(crop_size),
        "--save_latest_freq",str(save_latest_freq),
        "--save_epoch_freq", "1",
        "--display_id",      "-1",
    ]
    print("Running fresh training:\n", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)


def train_resume(
    dataroot: str,
    name: str = "pix2pix_main",
    model: str = "pix2pix",
    direction: str = "AtoB",
    epoch_count: int = 1,
    n_epochs: int = 1,          # placeholder, will override below
    n_epochs_decay: int = 0,
    batch_size: int = 1,
    gpu_ids: int = -1,
    load_size: int = 512,
    crop_size: int = 512,
    save_latest_freq: int = 1,
):
    """
    Resume training using images in <dataroot>/train.
    Continues from the latest checkpoint for exactly RESUME_EPOCHS more epochs.
    """
    train_dir = os.path.join(dataroot, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    # find last completed epoch N
    ckpt_dir = os.path.join("checkpoints", name)
    pattern = re.compile(r"^(\d+)_net_G\.pth$")
    epochs = []
    for fname in os.listdir(ckpt_dir):
        m = pattern.match(fname)
        if m:
            epochs.append(int(m.group(1)))
    if not epochs:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}. Run fresh first.")
    last = max(epochs)

    # now set to train from last+1 through last+RESUME_EPOCHS
    start = last + 1
    end = last + RESUME_EPOCHS

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--dataroot",        dataroot,
        "--name",            name,
        "--model",           model,
        "--direction",       direction,
        "--continue_train",
        "--epoch_count",     str(start),
        "--n_epochs",        str(end),
        "--n_epochs_decay",  str(0),
        "--batch_size",      str(batch_size),
        "--gpu_ids",         str(gpu_ids),
        "--load_size",       str(load_size),
        "--crop_size",       str(crop_size),
        "--save_latest_freq",str(save_latest_freq),
        "--save_epoch_freq", "1",
        "--display_id",      "-1",
    ]
    print(f"Resuming training (+{RESUME_EPOCHS} epochs):\n", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    print(
        """
Choose training mode:
  [1] Train fresh (start a new run)
  [2] Resume training (+5 epochs)
"""
    )
    choice = input("Enter 1 or 2: ").strip()

    DATA_ROOT = "data"
    EXP_NAME  = "pix2pix_main"

    if choice == "1":
        train_fresh(dataroot=DATA_ROOT, name=EXP_NAME)

    elif choice == "2":
        train_resume(dataroot=DATA_ROOT, name=EXP_NAME)

    else:
        print("Invalid choice. Exiting.")
