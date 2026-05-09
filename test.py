"""
Evaluation script — loads a trained checkpoint and reports D4RL normalized scores.

Usage:
  python test.py --model bc  --ckpt models/best_model_bc.pth  --device cuda:0
  python test.py --model dt  --ckpt models/best_model_dt.pth  --device cuda:0
  python test.py --model odt --ckpt models/best_model_odt.pth --device cuda:0
"""

import argparse
import numpy as np
import torch

from src.model import BehaviorCloning, DecisionTransformer, OnlineDecisionTransformer
from src.dataloader import ENV_META
from src.utils import set_seed, evaluate_bc, evaluate_dt, d4rl_normalized_score


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained RL model")
    p.add_argument("--model", type=str, required=True,
                   choices=["bc", "dt", "odt"])
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--dataset", type=str, default="hopper-medium-replay-v2")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--target_return", type=float, default=3600.0)

    # architecture (must match training)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--context_len", type=int, default=20)
    p.add_argument("--dropout", type=float, default=0.1)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    meta = ENV_META[args.dataset]
    state_dim, act_dim = meta["state_dim"], meta["act_dim"]

    print(f"Device: {device}")
    print(f"Model:  {args.model.upper()}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Environment: {meta['env_name']}")
    print()

    # build model
    if args.model == "bc":
        model = BehaviorCloning(
            state_dim, act_dim,
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            max_len=args.context_len, dropout=args.dropout,
        )
    elif args.model == "dt":
        model = DecisionTransformer(
            state_dim, act_dim,
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            max_len=args.context_len, dropout=args.dropout,
        )
    elif args.model == "odt":
        model = OnlineDecisionTransformer(
            state_dim, act_dim,
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            max_len=args.context_len, dropout=args.dropout,
        )

    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print(f"Evaluating over {args.num_episodes} episodes...\n")

    if args.model == "bc":
        mean_ret, std_ret = evaluate_bc(
            model, meta["env_name"], device,
            num_episodes=args.num_episodes,
            max_ep_len=1000, context_len=args.context_len,
            state_dim=state_dim,
        )
    else:
        mean_ret, std_ret = evaluate_dt(
            model, meta["env_name"], device,
            target_return=args.target_return,
            num_episodes=args.num_episodes,
            max_ep_len=1000, context_len=args.context_len,
            state_dim=state_dim, act_dim=act_dim,
        )

    norm_score = d4rl_normalized_score(args.dataset, mean_ret)

    print("=" * 50)
    print(f"Model:                {args.model.upper()}")
    print(f"Average Return:       {mean_ret:.2f} +/- {std_ret:.2f}")
    print(f"D4RL Normalized Score: {norm_score:.2f}")
    print("=" * 50)

    # also evaluate at multiple target returns for DT/ODT
    if args.model in ["dt", "odt"]:
        print("\nSweep over target returns:")
        print(f"{'Target RTG':>12} | {'Mean Return':>12} | {'D4RL Score':>10}")
        print("-" * 40)
        for tr in [1800, 2400, 3000, 3600]:
            m, s = evaluate_dt(
                model, meta["env_name"], device,
                target_return=tr,
                num_episodes=args.num_episodes // 2,
                max_ep_len=1000, context_len=args.context_len,
                state_dim=state_dim, act_dim=act_dim,
            )
            n = d4rl_normalized_score(args.dataset, m)
            print(f"{tr:>12.0f} | {m:>12.1f} | {n:>10.1f}")


if __name__ == "__main__":
    main()
