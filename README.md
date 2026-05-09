# Offline Reinforcement Learning via Sequence Modeling

Implements and compares three transformer-based offline RL algorithms on the D4RL HalfCheetah/Hopper benchmark: Behavior Cloning (BC), Decision Transformer (DT), and Online Decision Transformer (ODT).

## Models

| Model | Description |
|-------|-------------|
| BC | Baseline transformer that imitates dataset actions |
| DT | Return-conditioned sequence model (Decision Transformer) |
| ODT | Stochastic DT with online fine-tuning |

## Dataset

D4RL `hopper-medium-replay-v2` — downloaded automatically via `dataset_setup.py`.

```bash
python dataset_setup.py --output data/
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.12 and a CUDA-capable GPU (CPU inference supported).

## Training

```bash
# Behavior Cloning
python train.py --model bc --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/

# Decision Transformer
python train.py --model dt --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/

# Online Decision Transformer
python train.py --model odt --epochs 50 --online_epochs 10 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
```

## Evaluation

```bash
python test.py --model dt --ckpt models/best_model_dt.pth --device cuda:0 --num_episodes 20
```

Expected D4RL normalized score: ~60-80 (DT).

## License

MIT — see [LICENSE](LICENSE)
