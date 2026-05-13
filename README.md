# offline-rl-sequence-modeling

Three transformer-based offline RL methods on the D4RL Hopper benchmark. Behavior Cloning, Decision Transformer, and Online Decision Transformer trained on `hopper-medium-replay-v2` with D4RL-normalized scoring. Multi-seed evaluation pipeline for all three.

## Hero numbers

| Metric | Value |
|---|---|
| Methods | 3 (BC, DT, ODT) |
| Dataset | D4RL hopper-medium-replay-v2, ~200K transitions |
| Seeds per method | Multi-seed eval scripts ship for BC, DT, ODT (see `eval_*_multiseed.py`) |
| Expected DT D4RL-normalized score | 60 to 80 on Hopper |
| Foundation baseline | `foundation_compare.py` benchmarks a frozen transformer with linear head |

## Three methods, one benchmark

| Model | What it learns | Where it lives |
|---|---|---|
| BC | Imitates dataset actions through a transformer policy head | `train.py --model bc` |
| DT | Return-conditioned sequence model. State, action, return-to-go tokens | `train.py --model dt` |
| ODT | Stochastic DT with an online fine-tune phase | `train.py --model odt --online_epochs 10` |

Each model uses the same dataloader and tokenization, so the comparison isolates the algorithmic difference.

## Repository layout

```
offline-rl-sequence-modeling/
  src/
    dataloader.py        D4RL trajectory loader, return-to-go tokenization
    model.py             Transformer policy with mode flags for BC, DT, ODT
    utils.py             D4RL normalized score, eval rollout, seed helpers
  dataset_setup.py       D4RL download wrapper
  train.py               Single-seed training entry, mode flag picks BC, DT, ODT
  train_improved.py      Tuned hyperparameters from the multi-seed sweep
  test.py                Episode rollout with deterministic policy
  eval_bc_multiseed.py   BC eval across seeds, returns mean and std
  eval_dt_multiseed.py   DT eval across seeds
  eval_odt_multiseed.py  ODT eval across seeds
  eval_bc_checkpoints.py BC checkpoint sweep, picks best epoch
  compare.py             Side-by-side bar chart of D4RL scores
  foundation_compare.py  Frozen-foundation baseline plus DT on top
  plot_bc_final.py       Final BC curves
  plot_bc_multiseed.py   BC seed-spread plot
  requirements.txt
```

## Train

```
python dataset_setup.py --output data/

python train.py --model bc  --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
python train.py --model dt  --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
python train.py --model odt --epochs 50 --online_epochs 10 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
```

## Evaluate

Single seed, single model:

```
python test.py --model dt --ckpt models/best_model_dt.pth --device cuda:0 --num_episodes 20
```

Multi-seed for fair comparison:

```
python eval_bc_multiseed.py  --ckpt_dir outputs/bc  --seeds 5
python eval_dt_multiseed.py  --ckpt_dir outputs/dt  --seeds 5
python eval_odt_multiseed.py --ckpt_dir outputs/odt --seeds 5
python compare.py
```

`compare.py` reads the eval JSON dumps and writes a D4RL-score bar chart with mean and std error bars.

## Foundation comparison

`foundation_compare.py` tests whether a frozen transformer foundation gives a useful feature space for offline RL. A frozen encoder plus a linear policy head is the lower bound. DT trained from scratch is the upper bound. The gap shows how much the sequence modeling buys over a static representation.

## Stack

Python 3.12, PyTorch, D4RL, numpy, matplotlib for plots.

## Hardware

CPU runs to completion on the BC model in a few hours. DT and ODT want CUDA. Reduce `--batch_size` and `--epochs` to fit smaller VRAM. The dataloader streams transitions, no full-dataset GPU residency required.

## License

MIT, see `LICENSE`.
