import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from gnn_ppo_isp import (
    load_germany50_from_tgz,
    ISPEnv,
    GNNActorCritic
)

############################################
# VISUALIZATION (STABLE VERSION)
############################################
def visualize_network(edge_index, num_nodes, utilizations=None,
                      epoch="TEST", save_dir="output"):

    os.makedirs(save_dir, exist_ok=True)

    G = nx.DiGraph()
    edges = edge_index.t().cpu().numpy()

    # Add edges with utilization
    for i, (u, v) in enumerate(edges):
        w = float(utilizations[i]) if utilizations is not None and i < len(utilizations) else 0.0
        G.add_edge(int(u), int(v), weight=w)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 9))

    # Nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=400,
        node_color="lightblue",
        alpha=0.9
    )

    # Edges
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos,
        edge_color=weights,
        edge_cmap=plt.cm.Reds,
        width=2.2,
        arrows=True,
        arrowsize=12,
        alpha=0.8
    )

    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(
        f"Germany50 – Learned Routing (GNN + PPO)",
        fontsize=14,
        fontweight="bold"
    )

    plt.axis("off")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"network_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[OK] Visualization saved to: {save_path}")


############################################
# EVALUATION
############################################
TGZ_PATH = "directed-germany50-DFN-aggregated-5min-over-1day-native (1).tgz"
MODEL_PATH = "output/gnn_ppo_final.pth"

def evaluate():
    print("Loading dataset...")
    edge_index, capacities, traffic_series, num_nodes = load_germany50_from_tgz(TGZ_PATH)
    env = ISPEnv(edge_index, capacities, traffic_series)

    print("Loading trained model...")
    node_dim = num_nodes * 2 + 1
    model = GNNActorCritic(
        node_dim=node_dim,
        hidden_dim=128,
        num_edges=edge_index.shape[1]
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    state, _ = env.reset()

    rewards = []
    max_utils = []

    print("Running evaluation...")
    for step in range(200):

        node_features = torch.tensor(state[:-1], dtype=torch.float32)
        node_features = node_features.view(num_nodes, 2)

        node_features = torch.cat([
            node_features,
            torch.zeros(num_nodes, node_dim - 2)
        ], dim=1)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            batch=torch.zeros(num_nodes, dtype=torch.long)
        )

        with torch.no_grad():
            action, _ = model(data)

        state, reward, done, _, info = env.step(action.squeeze().numpy())

        rewards.append(reward)
        max_utils.append(info["max_util"])

        if done:
            break

    print("\n=== GNN + PPO EVALUATION RESULTS ===")
    print(f"Average Reward        : {np.mean(rewards):.4f}")
    print(f"Average Max Util (MLU): {np.mean(max_utils):.4f}")

    print("\nGenerating visualization...")
    visualize_network(
        edge_index=edge_index,
        num_nodes=num_nodes,
        utilizations=info["utilization"],
        epoch="TEST"
    )


############################################
# MAIN
############################################
if __name__ == "__main__":
    evaluate()
