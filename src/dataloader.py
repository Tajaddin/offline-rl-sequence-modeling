"""
Data loading utilities for D4RL offline RL datasets (HDF5 format).

Downloads and loads the hopper-medium-replay-v2 dataset without requiring
the d4rl package (which has Windows compatibility issues).
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────

DATASET_URLS = {
    "hopper-medium-replay-v2": (
        "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/"
        "hopper_medium_replay-v2.hdf5"
    ),
}

# D4RL reference scores for normalized scoring
REF_SCORES = {
    "hopper-medium-replay-v2": {
        "random": -20.272305,
        "expert": 3234.3,
    },
}

ENV_META = {
    "hopper-medium-replay-v2": {
        "env_name": "Hopper-v4",
        "state_dim": 11,
        "act_dim": 3,
    },
}


# ────────────────────────────────────────────────────────────────
# Raw data loading
# ────────────────────────────────────────────────────────────────

def load_d4rl_dataset(data_dir, dataset_name="hopper-medium-replay-v2"):
    """Load an HDF5 dataset and return a dict of numpy arrays."""
    path = os.path.join(data_dir, f"{dataset_name}.hdf5")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Run dataset_setup.py first to download it."
        )

    with h5py.File(path, "r") as f:
        data = {
            "observations": f["observations"][:],
            "actions": f["actions"][:],
            "rewards": f["rewards"][:],
            "terminals": f["terminals"][:],
            "timeouts": f["timeouts"][:] if "timeouts" in f else np.zeros_like(f["terminals"][:]),
        }
    return data


def segment_into_trajectories(data):
    """Split flat arrays into a list of trajectory dicts."""
    done = np.logical_or(data["terminals"], data["timeouts"])
    split_indices = np.where(done)[0] + 1

    trajectories = []
    start = 0
    for end in split_indices:
        traj = {k: v[start:end] for k, v in data.items()}
        # compute returns-to-go
        rewards = traj["rewards"]
        rtg = np.zeros_like(rewards)
        rtg[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            rtg[t] = rewards[t] + rtg[t + 1]
        traj["returns_to_go"] = rtg
        trajectories.append(traj)
        start = end

    return trajectories


# ────────────────────────────────────────────────────────────────
# PyTorch Datasets
# ────────────────────────────────────────────────────────────────

class BCDataset(Dataset):
    """Dataset for Behavior Cloning: returns (states, actions) windows."""

    def __init__(self, trajectories, context_len=20):
        self.context_len = context_len
        self.trajectories = trajectories

        # build index: (traj_idx, start_within_traj)
        self.indices = []
        for i, traj in enumerate(trajectories):
            traj_len = len(traj["observations"])
            for start in range(traj_len):
                self.indices.append((i, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start = self.indices[idx]
        traj = self.trajectories[traj_idx]
        traj_len = len(traj["observations"])

        # get window [start - context_len + 1 : start + 1]
        end = start + 1
        begin = max(0, end - self.context_len)

        states = traj["observations"][begin:end].astype(np.float32)
        actions = traj["actions"][begin:end].astype(np.float32)

        # pad if shorter than context_len
        pad_len = self.context_len - states.shape[0]
        if pad_len > 0:
            states = np.concatenate([np.zeros((pad_len, states.shape[1]), dtype=np.float32), states])
            actions = np.concatenate([np.zeros((pad_len, actions.shape[1]), dtype=np.float32), actions])

        return torch.tensor(states), torch.tensor(actions)


class DTDataset(Dataset):
    """Dataset for Decision Transformer / ODT:
    returns (states, actions, returns_to_go, timesteps) windows."""

    def __init__(self, trajectories, context_len=20):
        self.context_len = context_len
        self.trajectories = trajectories

        self.indices = []
        for i, traj in enumerate(trajectories):
            traj_len = len(traj["observations"])
            for start in range(traj_len):
                self.indices.append((i, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start = self.indices[idx]
        traj = self.trajectories[traj_idx]
        traj_len = len(traj["observations"])

        end = start + 1
        begin = max(0, end - self.context_len)

        states = traj["observations"][begin:end].astype(np.float32)
        actions = traj["actions"][begin:end].astype(np.float32)
        rtg = traj["returns_to_go"][begin:end].astype(np.float32).reshape(-1, 1)
        timesteps = np.arange(begin, end, dtype=np.int64)

        seq_len = states.shape[0]
        pad_len = self.context_len - seq_len
        if pad_len > 0:
            states = np.concatenate([np.zeros((pad_len, states.shape[1]), dtype=np.float32), states])
            actions = np.concatenate([np.zeros((pad_len, actions.shape[1]), dtype=np.float32), actions])
            rtg = np.concatenate([np.zeros((pad_len, 1), dtype=np.float32), rtg])
            timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int64), timesteps])

        return (
            torch.tensor(states),
            torch.tensor(actions),
            torch.tensor(rtg),
            torch.tensor(timesteps),
        )


def get_data_stats(trajectories):
    """Compute dataset statistics for normalization and reporting."""
    all_returns = [traj["rewards"].sum() for traj in trajectories]
    all_lengths = [len(traj["rewards"]) for traj in trajectories]
    return {
        "num_trajectories": len(trajectories),
        "num_transitions": sum(all_lengths),
        "avg_return": np.mean(all_returns),
        "std_return": np.std(all_returns),
        "max_return": np.max(all_returns),
        "min_return": np.min(all_returns),
        "avg_length": np.mean(all_lengths),
    }
