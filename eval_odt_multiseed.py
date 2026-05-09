"""
Multi-seed evaluation for ODT. Use --ckpt best_model_odt.pth (offline-only)
or --ckpt best_model_odt_online.pth (after online fine-tune).
"""
import argparse, json, os, numpy as np, torch
from src.model import OnlineDecisionTransformer
from src.dataloader import ENV_META
from src.utils import set_seed, evaluate_dt, d4rl_normalized_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/best_model_odt_online.pth")
    p.add_argument("--dataset", default="hopper-medium-replay-v2")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[1, 7, 13, 21, 42, 99, 123, 2024])
    p.add_argument("--episodes_per_seed", type=int, default=25)
    p.add_argument("--target_return", type=float, default=3600.0)
    p.add_argument("--context_len", type=int, default=20)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--out", default="outputs/odt_final_eval.json")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    meta = ENV_META[args.dataset]
    state_dim, act_dim = meta["state_dim"], meta["act_dim"]

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Target return: {args.target_return}")
    print(f"Seeds: {args.seeds}  Episodes per seed: {args.episodes_per_seed}\n")

    all_returns, all_norms, per_seed = [], [], []
    for seed in args.seeds:
        set_seed(seed)
        model = OnlineDecisionTransformer(
            state_dim, act_dim,
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            max_len=args.context_len, dropout=args.dropout,
        )
        model.load_state_dict(
            torch.load(args.ckpt, map_location=device, weights_only=True))
        model.to(device).eval()

        mean_ret, std_ret = evaluate_dt(
            model, meta["env_name"], device,
            target_return=args.target_return,
            num_episodes=args.episodes_per_seed,
            max_ep_len=1000, context_len=args.context_len,
            state_dim=state_dim, act_dim=act_dim,
        )
        norm = d4rl_normalized_score(args.dataset, mean_ret)
        print(f"seed {seed:>4}  return={mean_ret:7.1f} +/- {std_ret:6.1f}  "
              f"D4RL={norm:5.2f}")
        per_seed.append({"seed": seed, "mean_return": float(mean_ret),
                         "std_return": float(std_ret), "d4rl": float(norm)})
        all_returns.append(mean_ret)
        all_norms.append(norm)

    all_returns = np.array(all_returns)
    all_norms = np.array(all_norms)
    summary = {
        "ckpt": args.ckpt, "target_return": args.target_return,
        "seeds": args.seeds, "episodes_per_seed": args.episodes_per_seed,
        "per_seed": per_seed,
        "return_mean": float(all_returns.mean()),
        "return_std": float(all_returns.std(ddof=1)),
        "d4rl_mean": float(all_norms.mean()),
        "d4rl_std": float(all_norms.std(ddof=1)),
        "d4rl_min": float(all_norms.min()),
        "d4rl_max": float(all_norms.max()),
    }
    print("\n" + "=" * 52)
    print(f"ODT multi-seed eval ({len(args.seeds)} seeds x "
          f"{args.episodes_per_seed} episodes)")
    print(f"Raw return : {summary['return_mean']:.1f} +/- "
          f"{summary['return_std']:.1f}")
    print(f"D4RL score : {summary['d4rl_mean']:.2f} +/- "
          f"{summary['d4rl_std']:.2f}  "
          f"(range {summary['d4rl_min']:.1f}-{summary['d4rl_max']:.1f})")
    print("=" * 52)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
