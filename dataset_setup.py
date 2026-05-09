"""
Download the D4RL hopper-medium-replay-v2 dataset (HDF5).

Usage:
    python dataset_setup.py --output data/
"""

import os
import sys
import argparse
import urllib.request


DATASET_URL = (
    "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/"
    "hopper_medium_replay-v2.hdf5"
)
DATASET_NAME = "hopper-medium-replay-v2.hdf5"


def download(url, dest_path):
    print(f"Downloading {url}")
    print(f"Saving to {dest_path}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {pct}%  ({mb:.1f} / {total_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=progress)
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Download D4RL dataset")
    parser.add_argument("--output", type=str, default="data/",
                        help="Directory to save the dataset")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    dest = os.path.join(args.output, DATASET_NAME)

    if os.path.exists(dest):
        print(f"Dataset already exists at {dest}. Skipping download.")
        return

    download(DATASET_URL, dest)

    # verify
    import h5py
    with h5py.File(dest, "r") as f:
        n = f["observations"].shape[0]
        s_dim = f["observations"].shape[1]
        a_dim = f["actions"].shape[1]
    print(f"Verified: {n} transitions, state_dim={s_dim}, act_dim={a_dim}")


if __name__ == "__main__":
    main()
