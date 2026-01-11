"""
Analyze & Compare Traffic Engineering Results
FIXED – ISP-only routing instead of ECMP
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from gnn_ppo_isp import (
    load_germany50_from_tgz,
    ISPEnv,
    GNNActorCritic
)

# =========================
# CONFIG
# =========================
TGZ_PATH = "directed-germany50-DFN-aggregated-5min-over-1day-native (1).tgz"
MODEL_PATH = "output/gnn_ppo_final.pth"
OUTPUT_DIR = "output"

EVAL_STEPS = 200
LOAD_FACTOR = 3.0   # 🔥 scale traffic manually
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# CORE METRIC
# =========================
def evaluate_policy(env, policy_fn, name):
    print(f"\nEvaluating {name}...")
    env.reset()
    max_utils = []

    for step in range(EVAL_STEPS):
        if step % 50 == 0:
            print(f"  Step {step}/{EVAL_STEPS}")

        try:
            action = policy_fn()
            _, _, done, _, info = env.step(action)
            max_utils.append(info["max_util"])

            if done:
                print(f"  Episode done at step {step}")
                env.reset()

        except Exception as e:
            print(f"  Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break

    if len(max_utils) == 0:
        print(f"  WARNING: No utilization data collected for {name}")
        return 0.0

    avg_mlu = float(np.mean(max_utils))
    print(f"{name:<15} Avg MLU = {avg_mlu:.4f} (from {len(max_utils)} steps)")
    return avg_mlu


# =========================
# BASELINES
# =========================
def isp_only_policy(env):
    """ISP-only routing: direct routing without any splitting"""

    # Build edge lookup if not exists (do this once)
    if not hasattr(env, 'edge_lookup'):
        print("  Building edge lookup table...")
        env.edge_lookup = {}
        edges_np = env.edge_index.cpu().numpy()
        for e in range(env.num_edges):
            src = edges_np[0, e]
            dst = edges_np[1, e]
            env.edge_lookup[(int(src), int(dst))] = e
        print(f"  Edge lookup built: {len(env.edge_lookup)} edges")

    action = np.zeros(env.num_edges, dtype=np.float32)

    # Get current demand from environment
    tm = env.traffic[env.t % len(env.traffic)]

    # Route each demand on direct link only
    routed_demands = 0
    for s in range(env.num_nodes):
        for d in range(env.num_nodes):
            if s == d:
                continue

            demand = tm[s, d]
            if demand <= 0:
                continue

            # Find direct edge from s to d
            key = (s, d)
            if key in env.edge_lookup:
                e = env.edge_lookup[key]
                action[e] = demand
                routed_demands += 1

    # Normalize to get routing weights
    total = action.sum()
    if total > 0:
        action = action / total
    else:
        # Fallback: uniform distribution
        action = np.ones(env.num_edges, dtype=np.float32) / env.num_edges

    return action


def random_policy(env):
    a = np.random.rand(env.num_edges)
    return a / a.sum()


# =========================
# GNN POLICY
# =========================
def gnn_policy_builder(env, edge_index, num_nodes):
    node_dim = num_nodes * 2 + 1

    model = GNNActorCritic(
        node_dim=node_dim,
        hidden_dim=128,
        num_edges=edge_index.shape[1]
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    def policy():
        state = env._get_state()

        node_feat = torch.tensor(state[:-1], dtype=torch.float32)
        node_feat = node_feat.view(num_nodes, 2)
        node_feat = torch.cat(
            [node_feat, torch.zeros(num_nodes, node_dim - 2)],
            dim=1
        )

        data = Data(
            x=node_feat,
            edge_index=edge_index,
            batch=torch.zeros(num_nodes, dtype=torch.long)
        )

        with torch.no_grad():
            action, _ = model(data)

        return action.squeeze().numpy()

    return policy


# =========================
# MAIN
# =========================
def main():
    print("=" * 70)
    print("FIXED TRAFFIC ENGINEERING EVALUATION")
    print("=" * 70)

    # ---- LOAD DATA ----
    edge_index, capacities, traffic_series, num_nodes = \
        load_germany50_from_tgz(TGZ_PATH)

    # 🔥 SCALE TRAFFIC HERE (FIX)
    traffic_series = [tm * LOAD_FACTOR for tm in traffic_series]

    env = ISPEnv(edge_index, capacities, traffic_series)

    results = {}

    results["ISP-only"] = evaluate_policy(
        env,
        lambda: isp_only_policy(env),
        "ISP-only"
    )

    results["Random"] = evaluate_policy(
        env,
        lambda: random_policy(env),
        "Random"
    )

    gnn_policy = gnn_policy_builder(env, edge_index, num_nodes)
    results["GNN + PPO"] = evaluate_policy(
        env,
        gnn_policy,
        "GNN + PPO"
    )

    # =========================
    # SAVE TABLE
    # =========================
    df = pd.DataFrame({
        "Method": results.keys(),
        "Avg Max Utilization ↓": results.values()
    })

    table_path = os.path.join(OUTPUT_DIR, "te_comparison_table.csv")
    df.to_csv(table_path, index=False)

    print("\nFINAL COMPARISON TABLE")
    print(df.to_string(index=False))
    print(f"\nSaved to {table_path}")

    # =========================
    # PLOT
    # =========================
    plot_path = os.path.join(OUTPUT_DIR, "te_comparison_bar.png")

    plt.figure(figsize=(8, 5))
    plt.bar(
        df["Method"],
        df["Avg Max Utilization ↓"],
        color=["blue", "red", "green"]
    )

    plt.axhline(0.8, linestyle="--", color="orange", label="80% capacity")
    plt.axhline(1.0, linestyle="--", color="red", label="100% capacity")

    plt.ylabel("Average Maximum Link Utilization")
    plt.title("Traffic Engineering Performance (Germany50)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()