"""
Compare BC checkpoints with multi-seed evaluation.
Identifies which epoch checkpoint truly performs best in env (not just lowest val loss).
"""
import json, os, numpy as np, torch
from src.model import BehaviorCloning
from src.dataloader import ENV_META
from src.utils import set_seed, evaluate_bc, d4rl_normalized_score

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET = "hopper-medium-replay-v2"
META = ENV_META[DATASET]
STATE_DIM, ACT_DIM = META["state_dim"], META["act_dim"]

CKPTS = [
    "outputs/best_model_bc.pth",
    "outputs/bc_epoch10.pth",
    "outputs/bc_epoch20.pth",
    "outputs/bc_epoch30.pth",
    "outputs/bc_epoch40.pth",
    "outputs/bc_epoch50.pth",
]
SEEDS = [1, 7, 13, 21, 42]
EP_PER_SEED = 10  # 10 eps × 5 seeds = 50 rollouts per ckpt — fast scan

def eval_ckpt(path):
    returns, norms = [], []
    for seed in SEEDS:
        set_seed(seed)
        m = BehaviorCloning(STATE_DIM, ACT_DIM, d_model=128, n_heads=4,
                            n_layers=3, max_len=20, dropout=0.1)
        m.load_state_dict(torch.load(path, map_location=DEVICE,
                                     weights_only=True))
        m.to(DEVICE).eval()
        mr, _ = evaluate_bc(m, META["env_name"], DEVICE,
                            num_episodes=EP_PER_SEED, max_ep_len=1000,
                            context_len=20, state_dim=STATE_DIM)
        returns.append(mr)
        norms.append(d4rl_normalized_score(DATASET, mr))
    return np.array(returns), np.array(norms)

results = {}
print(f"Scanning {len(CKPTS)} checkpoints, {len(SEEDS)} seeds × "
      f"{EP_PER_SEED} eps each\n")
print(f"{'checkpoint':<35} {'D4RL mean':>10} {'std':>6} {'min':>6} {'max':>6}")
print("-" * 70)
for c in CKPTS:
    if not os.path.exists(c):
        print(f"{c:<35} MISSING")
        continue
    r, n = eval_ckpt(c)
    results[c] = {
        "return_mean": float(r.mean()), "return_std": float(r.std(ddof=1)),
        "d4rl_mean": float(n.mean()), "d4rl_std": float(n.std(ddof=1)),
        "d4rl_min": float(n.min()), "d4rl_max": float(n.max()),
        "per_seed_d4rl": [float(x) for x in n],
    }
    print(f"{c:<35} {n.mean():>10.2f} {n.std(ddof=1):>6.2f} "
          f"{n.min():>6.1f} {n.max():>6.1f}")

best = max(results, key=lambda k: results[k]["d4rl_mean"])
print(f"\nBest checkpoint by D4RL mean: {best} "
      f"({results[best]['d4rl_mean']:.2f} ± {results[best]['d4rl_std']:.2f})")

with open("outputs/bc_checkpoint_scan.json", "w") as f:
    json.dump({"results": results, "best_ckpt": best,
               "seeds": SEEDS, "episodes_per_seed": EP_PER_SEED}, f, indent=2)
print("Saved -> outputs/bc_checkpoint_scan.json")
