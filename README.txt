
# README

------------------------------------------------------------

## 1. Project Overview

Project Title: Sequence Modeling for Offline Reinforcement Learning

Model Type:
Transformer (Causal GPT-style)

Objective:
Offline Reinforcement Learning via Sequence Modeling — framing RL as a
conditional sequence prediction problem using transformer architectures.

Models Implemented:
  1. Behavior Cloning (BC) — baseline transformer that imitates dataset actions
  2. Decision Transformer (DT) — return-conditioned sequence model
  3. Online Decision Transformer (ODT) — stochastic DT with online fine-tuning

Dataset Used:
D4RL hopper-medium-replay-v2
http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5

Expected test evaluation for sanity check: D4RL Normalized Score ~= 60-80 (DT)

------------------------------------------------------------

## 2. Repository Structure

```
Main/
  train.py                  # Training script (all 3 models)
  test.py                   # Evaluation script
  dataset_setup.py          # Downloads D4RL dataset
  requirements.txt          # Python dependencies
  README.txt                # This file
  src/
    __init__.py
    model.py                # BC, DT, ODT model architectures
    dataloader.py           # Dataset loading and PyTorch Datasets
    utils.py                # Evaluation and utility functions
  models/                   # Model checkpoints go here
  data/                     # Downloaded dataset goes here
  outputs/                  # Training outputs (losses, plots, checkpoints)
```

------------------------------------------------------------

## 3. Dataset (OPTION A — PUBLIC DATASET SPLITS)

Dataset Link:
http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5

Where to place the downloaded dataset:
```
data/
  hopper-medium-replay-v2.hdf5
```

Download command:
```
python dataset_setup.py --output data/
```

------------------------------------------------------------

## 4. Model Checkpoint

Where to place the checkpoint after downloading:
```
models/
  best_model_bc.pth
  best_model_dt.pth
  best_model_odt.pth
  best_model_odt_online.pth
```

------------------------------------------------------------

## 5. Requirements (Dependencies)

Python Version: 3.12

How to install all dependencies:

Using pip:
```
pip install -r requirements.txt
```

Using conda:
```
conda create -n dlproj python=3.12
conda activate dlproj
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Note: PyTorch must be installed with CUDA support for GPU acceleration.

------------------------------------------------------------

## 6. Running the Test Script

Command to run testing:
```
python test.py --model dt --ckpt models/best_model_dt.pth --device cuda:0 --num_episodes 20

python test.py --model bc --ckpt models/best_model_bc.pth --device cuda:0 --num_episodes 20

python test.py --model odt --ckpt models/best_model_odt_online.pth --device cuda:0 --num_episodes 20
```

------------------------------------------------------------

## 7. Running the Training Script

Train Behavior Cloning (baseline):
```
python train.py --model bc --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
```

Train Decision Transformer:
```
python train.py --model dt --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
```

Train Online Decision Transformer (offline + online):
```
python train.py --model odt --epochs 50 --online_epochs 10 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
```

Optional arguments:
- `--seed 42`          (reproducibility)
- `--save_every 10`    (checkpoint frequency)
- `--eval_every 5`     (environment evaluation frequency)
- `--target_return 3600` (target return for DT/ODT)
- `--context_len 20`   (transformer context window)

------------------------------------------------------------

## 8. Submission Checklist

- [x] Dataset provided using Option A and placed correctly.
- [x] Model checkpoint instructions included.
- [x] requirements.txt generated and Python version specified.
- [x] Test command works.
- [x] Train command works.

------------------------------------------------------------
