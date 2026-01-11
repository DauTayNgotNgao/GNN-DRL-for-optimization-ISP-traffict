"""
Main Training Script with Comprehensive Logging
Run this to train GNN-PPO on full Germany50 dataset
"""

import os
import sys
import json
import time
from datetime import datetime
import argparse

# Import from main script
try:
    from gnn_ppo_isp import (
        load_germany50_from_tgz,
        ISPEnv,
        GNNActorCritic,
        PPO,
        visualize_network
    )
except ImportError:
    print("Error: Please ensure gnn_ppo_isp.py is in the same directory")
    sys.exit(1)

import torch
import numpy as np


class TrainingLogger:
    """Enhanced logger for training"""

    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.json')
        self.text_log = os.path.join(log_dir, f'training_{timestamp}.txt')

        self.logs = {
            'start_time': timestamp,
            'config': {},
            'epochs': [],
            'rewards': [],
            'max_utils': [],
            'avg_utils': [],
            'losses': [],
            'policy_losses': [],
            'value_losses': [],
            'episodes': []
        }

        # Text logging
        self.text_file = open(self.text_log, 'w')
        self.log_text(f"Training started at {timestamp}")
        self.log_text("=" * 70)

    def log_config(self, config):
        """Log training configuration"""
        self.logs['config'] = config
        self.save()

        self.log_text("\nTRAINING CONFIGURATION:")
        for key, value in config.items():
            self.log_text(f"  {key}: {value}")
        self.log_text("=" * 70 + "\n")

    def log_text(self, message):
        """Log text message"""
        print(message)
        self.text_file.write(message + '\n')
        self.text_file.flush()

    def log_epoch(self, epoch, reward, max_util, avg_util, loss, policy_loss, value_loss):
        """Log epoch data"""
        self.logs['epochs'].append(epoch)
        self.logs['rewards'].append(float(reward))
        self.logs['max_utils'].append(float(max_util))
        self.logs['avg_utils'].append(float(avg_util))
        self.logs['losses'].append(float(loss))
        self.logs['policy_losses'].append(float(policy_loss))
        self.logs['value_losses'].append(float(value_loss))

        if epoch % 10 == 0:
            self.save()

    def log_episode(self, episode, total_reward, avg_max_util):
        """Log episode completion"""
        self.logs['episodes'].append({
            'episode': episode,
            'total_reward': float(total_reward),
            'avg_max_util': float(avg_max_util)
        })
        self.save()

    def save(self):
        """Save logs to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def close(self):
        """Close logger"""
        self.log_text("\n" + "=" * 70)
        self.log_text(f"Training ended at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.text_file.close()
        self.save()


def train_with_logging(args):
    """Main training with comprehensive logging"""

    # Create logger
    logger = TrainingLogger(log_dir=args.log_dir)

    # Log configuration
    config = {
        'dataset': args.dataset,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'hidden_dim': args.hidden_dim,
        'update_interval': args.update_interval,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    logger.log_config(config)

    # Load data
    logger.log_text("Loading dataset...")
    start_time = time.time()

    try:
        edge_index, capacities, traffic_series, num_nodes = load_germany50_from_tgz(args.dataset)
        load_time = time.time() - start_time
        logger.log_text(f"Dataset loaded in {load_time:.2f} seconds")
    except Exception as e:
        logger.log_text(f"ERROR loading dataset: {e}")
        logger.close()
        return

    # Create environment
    env = ISPEnv(edge_index, capacities, traffic_series)
    logger.log_text(f"\nEnvironment created:")
    logger.log_text(f"  Nodes: {num_nodes}")
    logger.log_text(f"  Edges: {edge_index.shape[1]}")
    logger.log_text(f"  Traffic matrices: {len(traffic_series)}")

    # Create model
    node_dim = num_nodes * 2 + 1
    model = GNNActorCritic(
        node_dim=node_dim,
        hidden_dim=args.hidden_dim,
        num_edges=edge_index.shape[1]
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.log_text(f"\nModel created: {total_params:,} parameters")

    # Create agent
    agent = PPO(model, lr=args.lr)

    # Training loop
    logger.log_text("\n" + "=" * 70)
    logger.log_text("STARTING TRAINING")
    logger.log_text("=" * 70 + "\n")

    best_reward = -float('inf')
    best_max_util = float('inf')

    state, _ = env.reset()
    episode_reward = 0
    episode_max_utils = []
    step_count = 0
    episode_count = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Prepare data
        node_features = torch.tensor(state[:-1], dtype=torch.float32)
        node_features = node_features.view(num_nodes, 2)
        node_features = torch.cat([
            node_features,
            torch.zeros(num_nodes, node_dim - 2)
        ], dim=1)

        from torch_geometric.data import Data
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
        episode_max_utils.append(info['max_util'])
        step_count += 1

        # Update model
        if step_count % args.update_interval == 0:
            loss, policy_loss, value_loss = agent.update(data)
        else:
            loss = policy_loss = value_loss = 0.0

        # Log epoch
        logger.log_epoch(
            epoch, reward, info['max_util'], info['avg_util'],
            loss, policy_loss, value_loss
        )

        # Periodic logging
        if epoch % 20 == 0:
            elapsed = time.time() - epoch_start
            logger.log_text(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.3f}s)")
            logger.log_text(f"  Reward: {reward:.4f} | Episode Total: {episode_reward:.4f}")
            logger.log_text(f"  Max Util: {info['max_util']:.4f} | Avg Util: {info['avg_util']:.4f}")

            if step_count % args.update_interval == 0:
                logger.log_text(f"  Loss: {loss:.4f} (P: {policy_loss:.4f}, V: {value_loss:.4f})")

            # Track best
            if reward > best_reward:
                best_reward = reward
                logger.log_text(f"  *** New best reward: {best_reward:.4f} ***")

                # Save best model
                best_model_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)

            if info['max_util'] < best_max_util:
                best_max_util = info['max_util']
                logger.log_text(f"  *** New best max util: {best_max_util:.4f} ***")

        # Visualization
        if epoch % 100 == 0 and epoch > 0:
            logger.log_text(f"\n  Saving visualization...")
            visualize_network(edge_index, num_nodes, info['utilization'], epoch, args.output_dir)

        # Episode end
        if done:
            episode_count += 1
            avg_max_util = np.mean(episode_max_utils)

            logger.log_text(f"\n{'=' * 70}")
            logger.log_text(f"Episode {episode_count} completed (epoch {epoch})")
            logger.log_text(f"  Total Reward: {episode_reward:.4f}")
            logger.log_text(f"  Avg Max Util: {avg_max_util:.4f}")
            logger.log_text(f"{'=' * 70}\n")

            logger.log_episode(episode_count, episode_reward, avg_max_util)

            # Reset
            state, _ = env.reset()
            episode_reward = 0
            episode_max_utils = []
        else:
            state = next_state

        # Save checkpoint
        if epoch > 0 and epoch % 500 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'best_reward': best_reward,
            }, checkpoint_path)
            logger.log_text(f"  Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)

    logger.log_text("\n" + "=" * 70)
    logger.log_text("TRAINING COMPLETED!")
    logger.log_text("=" * 70)
    logger.log_text(f"Best reward: {best_reward:.4f}")
    logger.log_text(f"Best max util: {best_max_util:.4f}")
    logger.log_text(f"Final model saved: {final_model_path}")
    logger.log_text(f"Best model saved: {os.path.join(args.output_dir, 'best_model.pth')}")
    logger.log_text(f"Logs saved: {logger.log_file}")

    logger.close()


def main():
    parser = argparse.ArgumentParser(description='Train GNN-PPO on Germany50 Dataset')

    parser.add_argument('--dataset', type=str,
                        default='directed-germany50-DFN-aggregated-5min-over-1day-native (1).tgz',
                        help='Path to dataset TGZ file')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for GNN')
    parser.add_argument('--update-interval', type=int, default=32,
                        help='Update model every N steps')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for models and visualizations')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for log files')

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 70)
    print("GNN + PPO Training on Full Germany50 Dataset")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Hidden Dim: {args.hidden_dim}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Log Dir: {args.log_dir}")
    print("=" * 70 + "\n")

    # Train
    try:
        train_with_logging(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()