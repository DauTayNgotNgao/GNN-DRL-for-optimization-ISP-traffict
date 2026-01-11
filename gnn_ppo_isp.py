"""
Complete GNN + PPO for ISP Traffic Engineering
Load ALL data from directed-germany50-DFN-aggregated-5min-over-1day-native.tgz
Requires: Python>=3.9, torch, torch-geometric, gymnasium, networkx, matplotlib
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
import os
import re
import tarfile
import glob

############################################
# 1. DATA LOADER FOR TGZ
############################################
def load_germany50_from_tgz(tgz_path):
    """
    Load complete Germany50 dataset from .tgz file
    """
    print(f"Loading dataset from {tgz_path}...")

    # Extract TGZ
    extract_dir = 'germany50_data'
    os.makedirs(extract_dir, exist_ok=True)

    print("Extracting TGZ file...")
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(extract_dir, filter='data')

    print("Extracted successfully!")

    # Find all traffic matrix files
    traffic_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in sorted(files):
            if file.endswith('.txt') and 'demandMatrix' in file:
                full_path = os.path.join(root, file)
                traffic_files.append(full_path)

    print(f"Found {len(traffic_files)} traffic matrix files")

    if not traffic_files:
        raise ValueError("No traffic matrix files found in TGZ!")

    print(f"\nParsing topology from {traffic_files[0]}...")

    try:
        edge_index, capacities, nodes_map = parse_topology_from_file(traffic_files[0])
        num_nodes = len(nodes_map)

        print(f"\nTopology parsed successfully:")
        print(f"  Nodes: {num_nodes}")
        print(f"  Edges: {edge_index.shape[1]}")

    except Exception as e:
        print(f"ERROR parsing topology: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Load all traffic matrices
    print("\nLoading all traffic matrices...")
    traffic_series = []

    for i, file_path in enumerate(traffic_files):
        try:
            tm = parse_traffic_matrix(file_path, nodes_map)
            if tm is not None and tm.sum() > 0:  # Only add non-empty matrices
                traffic_series.append(tm)

            if (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1}/{len(traffic_files)} files ({len(traffic_series)} valid)...")
        except Exception as e:
            print(f"  Warning: Failed to load {file_path}: {e}")

    if len(traffic_series) == 0:
        raise ValueError("No valid traffic matrices loaded!")

    print(f"\nSuccessfully loaded {len(traffic_series)} traffic matrices")
    print(f"Network: {num_nodes} nodes, {edge_index.shape[1]} edges")

    # Validate edge_index shape
    assert edge_index.shape[0] == 2, f"edge_index should have shape [2, num_edges], got {edge_index.shape}"
    assert edge_index.shape[1] > 0, "No edges in edge_index"

    return edge_index, capacities, traffic_series, num_nodes


def parse_topology_from_file(file_path):
    """Parse topology from SNDlib file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract nodes - FIXED: Match until closing ) on new line
    nodes_match = re.search(r'NODES\s*\((.*?)\n\)', content, re.DOTALL)
    nodes_map = {}
    node_list = []

    if nodes_match:
        nodes_section = nodes_match.group(1)
        for line in nodes_section.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '(' in line:
                # Match: NodeName ( longitude latitude )
                match = re.match(r'(\w+)\s*\(', line)
                if match:
                    node_id = match.group(1)
                    if node_id not in nodes_map:  # Avoid duplicates
                        nodes_map[node_id] = len(node_list)
                        node_list.append(node_id)

    num_nodes = len(node_list)

    if num_nodes == 0:
        raise ValueError("No nodes found! Check file format.")

    print(f"Found {num_nodes} nodes")
    if num_nodes <= 10:
        print(f"  Nodes: {node_list}")
    else:
        print(f"  First 10 nodes: {node_list[:10]}")
        print(f"  Last 10 nodes: {node_list[-10:]}")

    # Create topology - full mesh for realistic routing
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])

    print(f"Created {len(edges)} edges for full mesh topology")

    # Convert to tensor - IMPORTANT: transpose to get correct shape [2, num_edges]
    if len(edges) == 0:
        raise ValueError("No edges created! Check node parsing.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"Edge index shape: {edge_index.shape}")

    # Verify shape
    assert edge_index.shape[0] == 2, f"Edge index should have 2 rows, got {edge_index.shape[0]}"
    assert edge_index.shape[1] == len(edges), f"Edge index should have {len(edges)} columns, got {edge_index.shape[1]}"

    # Capacities (100 Mbps per link as default)
    num_edges = edge_index.shape[1]
    capacities = np.ones(num_edges, dtype=np.float32) * 100.0

    return edge_index, capacities, nodes_map


def parse_traffic_matrix(file_path, nodes_map):
    """Parse traffic matrix from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  Warning: Could not read {file_path}: {e}")
        return None

    num_nodes = len(nodes_map)
    traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Extract demands - FIXED: Match until closing ) on new line
    demands_match = re.search(r'DEMANDS\s*\((.*?)\n\)', content, re.DOTALL)
    if demands_match:
        demands_count = 0
        demands_section = demands_match.group(1)

        for line in demands_section.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '(' in line:
                # Match: demand_id ( source target ) routing_unit demand_value max_path
                # Example: Bremen_Nuernberg ( Bremen Nuernberg ) 1 0.045579 UNLIMITED
                match = re.match(r'\S+\s*\(\s*(\w+)\s+(\w+)\s*\)\s+\d+\s+([\d.]+)', line)
                if match:
                    source = match.group(1)
                    target = match.group(2)
                    value = float(match.group(3))

                    if source in nodes_map and target in nodes_map:
                        src_idx = nodes_map[source]
                        tgt_idx = nodes_map[target]
                        traffic_matrix[src_idx, tgt_idx] = value
                        demands_count += 1

        if demands_count == 0:
            print(f"  Warning: No demands parsed from {os.path.basename(file_path)}")
    else:
        print(f"  Warning: No DEMANDS section in {os.path.basename(file_path)}")

    return traffic_matrix


############################################
# 2. ENVIRONMENT
############################################
class ISPEnv(gym.Env):
    """ISP Traffic Engineering Environment"""

    def __init__(self, edge_index, capacities, traffic_series):
        super().__init__()
        self.edge_index = edge_index
        self.capacities = np.array(capacities, dtype=np.float32)
        self.traffic = traffic_series
        self.t = 0
        self.episode = 0
        self.num_edges = edge_index.shape[1]
        self.num_nodes = int(edge_index.max().item()) + 1

        # Action: routing weights for each edge
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_edges,),
            dtype=np.float32
        )

        # State: node features + time
        self.observation_space = gym.spaces.Box(
            low=0, high=1e9,
            shape=(self.num_nodes + 1,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Start from random position in traffic series
        self.t = np.random.randint(0, max(1, len(self.traffic) - 100))
        self.episode += 1
        return self._get_state(), {}

    def step(self, action):
        if self.t >= len(self.traffic):
            self.t = 0

        tm = self.traffic[self.t]

        # Route traffic
        utilization = self._route(tm, action)

        # Reward: negative MLU (Maximum Link Utilization) reward = - link co ti le cao nhat
        max_util = np.max(utilization)
        avg_util = np.mean(utilization)

        # Multi-objective reward
        reward = -max_util - 0.1 * avg_util  # Penalize both max and average

        self.t += 1
        done = (self.t % 100 == 0)  # Episode length = 100 steps

        info = {
            'utilization': utilization,
            'max_util': max_util,
            'avg_util': avg_util
        }

        return self._get_state(), reward, done, False, info

    def _get_state(self):
        if self.t >= len(self.traffic):
            self.t = 0

        tm = self.traffic[self.t]

        # Node features: total incoming/outgoing traffic
        node_in_traffic = tm.sum(axis=0).astype(np.float32)
        node_out_traffic = tm.sum(axis=1).astype(np.float32)

        # Normalize
        max_traffic = max(node_in_traffic.max(), node_out_traffic.max()) + 1e-8
        node_in_traffic = node_in_traffic / max_traffic
        node_out_traffic = node_out_traffic / max_traffic

        # State = [in_traffic, out_traffic, time_index]
        state = np.concatenate([
            node_in_traffic,
            node_out_traffic,
            [float(self.t) / len(self.traffic)]
        ])

        return state

    def _route(self, tm, weights):
        """
        Route traffic using weights
        Implements simplified traffic engineering
        """
        #khởi tạo tất cả các link = 0
        flow = np.zeros(len(self.capacities), dtype=np.float32)

        # Get source-destination pairs with traffic
        edges_np = self.edge_index.cpu().numpy()

        # For each source-destination pair with traffic
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src != dst and tm[src, dst] > 0:
                    #gán traffic
                    demand = tm[src, dst]

                    # Find edges that can carry this traffic
                    # Simplified: distribute based on edge weights
                    relevant_edges = []
                    for e_idx in range(len(edges_np[0])):
                        edge_src = edges_np[0, e_idx]
                        edge_dst = edges_np[1, e_idx]

                        # Duyệt tất cả đường khả thi cho việc đi từ A đến B
                        if edge_src == src or edge_dst == dst:
                            relevant_edges.append(e_idx)

                    if relevant_edges:
                        # Normalize weights for relevant edges
                        edge_weights = np.array([weights[e] for e in relevant_edges])
                        edge_weights = (edge_weights + 1.0) / 2.0  # [0, 1]
                        weight_sum = edge_weights.sum() + 1e-8
                        edge_weights = edge_weights / weight_sum

                        # Distribute traffic
                        for i, e_idx in enumerate(relevant_edges):
                            flow[e_idx] += demand * edge_weights[i]

        # Calculate utilization
        utilization = flow / (self.capacities + 1e-8) #luong traffic / tong dung luong
        return np.clip(utilization, 0, 10)  # Clip to prevent overflow


############################################
# 3. GNN ACTOR-CRITIC
############################################
class GNNActorCritic(nn.Module):
    """GNN-based Actor-Critic Network"""

    def __init__(self, node_dim, hidden_dim, num_edges):
        super().__init__()

        # GNN encoder
        self.gnn1 = GATConv(node_dim, hidden_dim, heads=4, concat=True)
        self.gnn2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.gnn3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_edges),
            nn.Tanh()
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # GNN layers
        x = torch.relu(self.gnn1(x, edge_index))
        x = torch.relu(self.gnn2(x, edge_index))
        x = torch.relu(self.gnn3(x, edge_index))

        # Global pooling
        pooled = global_mean_pool(x, batch)

        # Outputs
        action = self.actor(pooled)
        value = self.critic(pooled)

        return action, value


############################################
# 4. PPO ALGORITHM
############################################
class PPO:
    """Proximal Policy Optimization"""

    def __init__(self, model, lr=3e-4, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Experience buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'dones': []
        }

    def store_transition(self, state, action, reward, value, done):
        """Store transition in buffer"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['dones'].append(done)

    def clear_buffer(self):
        """Clear experience buffer"""
        for key in self.buffer:
            self.buffer[key] = []

    def update(self, data):
        """Update model with collected experience"""
        if len(self.buffer['rewards']) == 0:
            return 0.0, 0.0, 0.0

        # Convert to tensors
        actions = torch.stack([torch.tensor(a, dtype=torch.float32)
                               for a in self.buffer['actions']])
        rewards = torch.tensor(self.buffer['rewards'], dtype=torch.float32)
        old_values = torch.stack([v for v in self.buffer['values']])

        # Compute returns (simple Monte Carlo)
        returns = rewards.clone()

        # Compute advantages
        advantages = returns - old_values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        action_pred, value_pred = self.model(data)

        # Expand predictions to match batch
        if action_pred.size(0) == 1 and len(actions) > 1:
            action_pred = action_pred.expand(len(actions), -1)
            value_pred = value_pred.expand(len(actions), -1)

        # Policy loss
        policy_loss = torch.mean((action_pred - actions) ** 2)

        # Value loss
        value_loss = torch.mean((value_pred.squeeze() - returns) ** 2)

        # Total loss
        loss = policy_loss + self.value_coef * value_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        # Clear buffer
        self.clear_buffer()

        return loss.item(), policy_loss.item(), value_loss.item()


############################################
# 5. VISUALIZATION
############################################
def visualize_network(edge_index, num_nodes, utilizations=None, epoch=0, save_dir='output'):
    """Visualize network state"""

    def visualize_network(edge_index, num_nodes, utilizations=None, epoch=0, save_dir='output'):
        """Visualize network state with stable colorbar"""

        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            import os
            import numpy as np

            os.makedirs(save_dir, exist_ok=True)

            # Create directed graph
            G = nx.DiGraph()
            edges = edge_index.t().cpu().numpy()

            # Subsample edges for visualization (avoid clutter)
            step = max(1, len(edges) // 200)
            edge_sample = edges[::step]

            if utilizations is not None:
                util_sample = np.asarray(utilizations)[::step]
            else:
                util_sample = None

            # Add edges
            for i, (u, v) in enumerate(edge_sample):
                if util_sample is not None:
                    G.add_edge(int(u), int(v), weixght=float(util_sample[i]))
                else:
                    G.add_edge(int(u), int(v))

            # Layout
            pos = nx.spring_layout(G, seed=42, k=3, iterations=30)

            # Create figure & axes (IMPORTANT)
            fig, ax = plt.subplots(figsize=(14, 10))

            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos,
                node_size=400,
                node_color='lightblue',
                alpha=0.9,
                ax=ax
            )

            # Draw edges
            if util_sample is not None:
                weights = [G[u][v]['weight'] for u, v in G.edges()]

                nx.draw_networkx_edges(
                    G, pos,
                    edge_color=weights,
                    edge_cmap=plt.cm.RdYlGn_r,
                    edge_vmin=0.0,
                    edge_vmax=1.0,
                    width=2.5,
                    arrows=True,
                    arrowsize=12,
                    alpha=0.7,
                    ax=ax
                )

                # Colorbar (FIXED)
                sm = plt.cm.ScalarMappable(
                    cmap=plt.cm.RdYlGn_r,
                    norm=plt.Normalize(vmin=0.0, vmax=1.0)
                )
                sm.set_array([])

                cbar = fig.colorbar(
                    sm,
                    ax=ax,
                    fraction=0.046,
                    pad=0.04
                )
                cbar.set_label('Link Utilization', fontsize=11)

            else:
                nx.draw_networkx_edges(
                    G, pos,
                    width=2.0,
                    arrows=True,
                    arrowsize=12,
                    alpha=0.8,
                    ax=ax
                )

            # Draw labels
            nx.draw_networkx_labels(
                G, pos,
                font_size=7,
                font_weight='bold',
                ax=ax
            )

            # Title
            ax.set_title(
                f'Germany50 Network - Epoch {epoch}\n'
                f'(Showing {len(edge_sample)}/{len(edges)} edges)',
                fontsize=14,
                fontweight='bold'
            )
            ax.axis('off')

            save_path = os.path.join(save_dir, f'network_epoch_{epoch}.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved visualization: {save_path}")

        except Exception as e:
            print(f"    Visualization failed: {e}")


############################################
# 6. TRAINING
############################################
def train():
    """Main training loop"""
    print("="*70)
    print("GNN + PPO for ISP Traffic Engineering - Full Dataset Training")
    print("="*70)

    # Load data
    tgz_path = 'directed-germany50-DFN-aggregated-5min-over-1day-native (1).tgz'

    if not os.path.exists(tgz_path):
        print(f"\nError: {tgz_path} not found!")
        print("Please place the .tgz file in the current directory.")
        return

    try:
        edge_index, capacities, traffic_series, num_nodes = load_germany50_from_tgz(tgz_path)
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create environment
    env = ISPEnv(edge_index, capacities, traffic_series)

    print(f"\n{'='*70}")
    print("Environment Setup:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Traffic matrices: {len(traffic_series)}")
    print(f"  Total training steps: {len(traffic_series)}")
    print(f"{'='*70}")

    # Create model
    node_dim = num_nodes * 2 + 1
    model = GNNActorCritic(
        node_dim=node_dim,
        hidden_dim=128,
        num_edges=edge_index.shape[1]
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created: {total_params:,} parameters")

    # Create agent
    agent = PPO(model, lr=1e-4)

    # Training
    print(f"\n{'='*70}")
    print("Starting Training")
    print(f"{'='*70}\n")

    num_epochs = 1000
    update_interval = 32  # Update every N steps
    best_reward = -float('inf')

    state, _ = env.reset()
    episode_reward = 0
    step_count = 0

    for epoch in range(num_epochs):
        # Prepare graph data
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

        # Get action
        with torch.no_grad():
            action, value = model(data)

        action_np = action.squeeze().cpu().numpy()

        # Step environment
        next_state, reward, done, _, info = env.step(action_np)

        # Store transition
        agent.store_transition(state, action_np, reward, value, done)

        episode_reward += reward
        step_count += 1

        # Update model
        if step_count % update_interval == 0:
            loss, policy_loss, value_loss = agent.update(data)
        else:
            loss = policy_loss = value_loss = 0.0

        # Logging
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Reward: {reward:.4f} | Episode Reward: {episode_reward:.4f}")
            print(f"  Max Util: {info['max_util']:.4f} | Avg Util: {info['avg_util']:.4f}")

            if step_count % update_interval == 0:
                print(f"  Loss: {loss:.4f} (Policy: {policy_loss:.4f}, Value: {value_loss:.4f})")

            if reward > best_reward:
                best_reward = reward
                print(f"  *** New best reward: {best_reward:.4f} ***")

        # Visualization
        if epoch % 100 == 0 and epoch > 0:
            print(f"\n  Generating visualization...")
            visualize_network(edge_index, num_nodes, info['utilization'], epoch)

        # Reset on done
        if done:
            print(f"\n  Episode finished. Total reward: {episode_reward:.4f}\n")
            state, _ = env.reset()
            episode_reward = 0
        else:
            state = next_state

    # Save model
    os.makedirs('output', exist_ok=True)
    model_path = 'output/gnn_ppo_final.pth'
    torch.save(model.state_dict(), model_path)

    print(f"\n{'='*70}")
    print("Training Completed!")
    print(f"{'='*70}")
    print(f"Best reward: {best_reward:.4f}")
    print(f"Model saved: {model_path}")
    print(f"Visualizations saved in: output/")


############################################
# 7. MAIN
############################################
if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()