# train.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from models.behavior_cloning import BehavioralCloning
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BC agent.")
    parser.add_argument("--env_name", type=str, default="BipedalWalker-v3", help="Gym environment name")
    parser.add_argument("--expert_data", type=str, required=True, help="Path to the expert data .npz file")
    parser.add_argument("--num_trajectories", type=int, default=50, help="Number of expert trajectories to use")
    parser.add_argument("--n_epoch", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--n", type=int, default=1, help="Window size (number of consecutive states)")
    parser.add_argument("--compare", type=float, default=5.0, help="Filtering threshold")
    parser.add_argument("--eval_frequency", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--num_test_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max_test_steps", type=int, default=500, help="Maximum steps per evaluation episode")
    parser.add_argument("--test_seeds", nargs='+', type=int, default=None, help="Seeds for evaluation episodes")
    parser.add_argument("--plot_rewards", action="store_true", help="Plot rewards after training")

    args = parser.parse_args()

    # Extract data size from expert data filename
    data_size = "unknown"
    expert_data_filename = os.path.basename(args.expert_data)
    match = re.search(r'expert_data_(\d+)', expert_data_filename)
    if match:
        data_size = match.group(1)
    
    # Create output directory with data size information
    model_name = f"bc_{args.env_name}_{data_size}data_{args.n_epoch}epochs"
    output_dir = os.path.join("outputs", model_name)
    os.makedirs(output_dir, exist_ok=True)

    bc_agent = BehavioralCloning(
        args.env_name,
        args.expert_data,
        args.num_trajectories,
        args.n_epoch,
        args.batch_size,
        args.lr,
        args.n,
        args.compare,
        args.eval_frequency,
        args.num_test_episodes,
        args.max_test_steps,
        args.test_seeds
    )

    loss_history, training_rewards = bc_agent.train()

    # Save model in the output directory
    model_path = os.path.join(output_dir, "model.pth")
    bc_agent.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Save training rewards
    rewards_path = os.path.join(output_dir, "training_rewards.npy")
    np.save(rewards_path, np.array(training_rewards))
    print(f"Training rewards saved to {rewards_path}")

    # Evaluate the model
    test_rewards = bc_agent.evaluate()
    
    # Save test rewards
    test_rewards_path = os.path.join(output_dir, "test_rewards.npy")
    np.save(test_rewards_path, np.array(test_rewards))
    print(f"Test rewards saved to {test_rewards_path}")

    # Plotting
    if args.plot_rewards:
        # Plot training rewards
        epochs, rewards = zip(*training_rewards)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, rewards, 'b-o')
        plt.xlabel('Epoch')
        plt.ylabel('Average Reward')
        plt.title(f'Training Rewards - {args.env_name} ({data_size} data)')
        plt.grid(True)
        fig_path = os.path.join(output_dir, "training_rewards.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Training rewards plot saved to {fig_path}")
        
        # Plot test rewards
        plt.figure(figsize=(10, 6))
        plt.plot(test_rewards, 'b-o')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Test Rewards - {args.env_name} ({data_size} data)')
        plt.grid(True)
        fig_path = os.path.join(output_dir, "test_rewards.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Test rewards plot saved to {fig_path}")

    # Print summary statistics
    print(f"Average training reward: {np.mean([r for _, r in training_rewards]):.2f}")
    print(f"Average test reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")