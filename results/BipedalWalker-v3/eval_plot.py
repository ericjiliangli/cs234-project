import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

# Define directories
BC_DIR = "bc-bipedal"
GAIL_DIR = "gail-bipedal"  # Current directory where GAIL outputs are
PPO_DIR = "PPO"          # New folder containing PPO trend data
OUTPUT_DIR = "result/comparison_plots"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_data_size(directory_name):
    """Extract data size from directory name"""
    # For BC directories
    if "bc_BipedalWalker" in directory_name:
        match = re.search(r'(\d+)data', directory_name)
        if match:
            return int(match.group(1))
    # For GAIL directories
    elif "gail-" in directory_name:
        match = re.search(r'gail-(\d+)_expert', directory_name)
        if match:
            return int(match.group(1))
    return None

def load_rewards_from_npz(directory):
    """Load training and evaluation rewards from NPZ files in a directory"""
    training_rewards_path = os.path.join(directory, "train_reward.np.npz")
    eval_rewards_path = os.path.join(directory, "eval_reward.np.npz")
    
    training_rewards = None
    eval_rewards = None
    
    if os.path.exists(training_rewards_path):
        data = np.load(training_rewards_path)
        if 'rewards' in data:
            training_rewards = data['rewards']
    
    if os.path.exists(eval_rewards_path):
        data = np.load(eval_rewards_path)
        if 'rewards' in data:
            eval_rewards = data['rewards']
    
    return training_rewards, eval_rewards

def smooth_data(data, window_size=10):
    """Calculate average and standard deviation over windows of data"""
    if len(data) <= window_size:
        return np.array([np.mean(data)]), np.array([np.std(data)])
    
    num_windows = len(data) // window_size
    means = np.zeros(num_windows)
    stds = np.zeros(num_windows)
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_data = data[start_idx:end_idx]
        means[i] = np.mean(window_data)
        stds[i] = np.std(window_data)
    
    return means, stds

def create_plots():
    """Create separate plots for BC+PPO and GAIL+PPO including the PPO trend"""
    # Collect data
    bc_data = {}  # {data_size: {"training": array, "eval": array}}
    gail_data = {}
    
    # Find all BC directories
    for bc_dir in glob.glob(f"{BC_DIR}/bc-outputs-bc_BipedalWalker-v3_*data_2000epochs"):
        data_size = extract_data_size(bc_dir)
        if data_size is not None:
            training_rewards, eval_rewards = load_rewards_from_npz(bc_dir)
            bc_data[data_size] = {
                "training": training_rewards,
                "eval": eval_rewards
            }
    
    # Find all GAIL directories
    for gail_dir in glob.glob(f"{GAIL_DIR}/gail-*_expert-BipedalWalker-v3"):
        data_size = extract_data_size(gail_dir)
        if data_size is not None:
            training_rewards, eval_rewards = load_rewards_from_npz(gail_dir)
            gail_data[data_size] = {
                "training": training_rewards,
                "eval": eval_rewards
            }
    
    # Load PPO trend data (assumed to be one trend)
    ppo_training, ppo_eval = load_rewards_from_npz(PPO_DIR)
    ppo_data = {
        "training": ppo_training,
        "eval": ppo_eval
    }
    
    # Create separate plots for BC+PPO with PPO trend
    create_bc_training_plot(bc_data, ppo_data)
    create_bc_eval_plot(bc_data, ppo_data)
    
    # Create separate plots for GAIL+PPO with PPO trend
    create_gail_training_plot(gail_data, ppo_data)
    create_gail_eval_plot(gail_data, ppo_data)

def create_bc_training_plot(bc_data, ppo_data):
    """Create plot for BC+PPO training rewards with smoothing and PPO trend"""
    plt.figure(figsize=(11, 6.5))
    data_sizes = sorted(list(bc_data.keys()))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    window_size = 50
    
    for i, data_size in enumerate(data_sizes):
        if bc_data[data_size]["training"] is not None:
            rewards = bc_data[data_size]["training"]
            means, stds = smooth_data(rewards, window_size)
            x_points = np.arange(len(means)) * window_size + window_size // 2
            color = colors[i % len(colors)]
            plt.plot(
                x_points, 
                means, 
                label=f"{data_size} expert data steps", 
                color=color,
                linestyle='-',
                linewidth=2
            )
            plt.fill_between(
                x_points, 
                means - stds, 
                means + stds, 
                alpha=0.2, 
                color=color
            )
    
    # Plot PPO trend if available
    if ppo_data["training"] is not None:
        ppo_means, ppo_stds = smooth_data(ppo_data["training"], window_size)
        x_points = np.arange(len(ppo_means)) * window_size + window_size // 2
        plt.plot(
            x_points, 
            ppo_means, 
            label="PPO Only", 
            color="black",
            linestyle='--',
            linewidth=2
        )
        plt.fill_between(
            x_points, 
            ppo_means - ppo_stds, 
            ppo_means + ppo_stds, 
            alpha=0.2, 
            color="black"
        )
    
    plt.title("BC+PPO Training Rewards by Expert Data Size (Averaged)")
    plt.xlabel(f"Training Steps (Window Size: {window_size})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/bc_training_rewards.png")
    plt.close()

def create_bc_eval_plot(bc_data, ppo_data):
    """Create plot for BC+PPO evaluation rewards with smoothing and PPO trend"""
    plt.figure(figsize=(11, 6.5))
    data_sizes = sorted(list(bc_data.keys()))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    window_size = 15  # Smaller window for eval data
    
    for i, data_size in enumerate(data_sizes):
        if bc_data[data_size]["eval"] is not None:
            rewards = bc_data[data_size]["eval"]
            means, stds = smooth_data(rewards, window_size)
            x_points = np.arange(len(means)) * window_size + window_size // 2
            color = colors[i % len(colors)]
            plt.plot(
                x_points, 
                means, 
                label=f"{data_size} expert data steps", 
                color=color,
                linestyle='-',
                linewidth=2
            )
            plt.fill_between(
                x_points, 
                means - stds, 
                means + stds, 
                alpha=0.2, 
                color=color
            )
    
    # Plot PPO trend if available
    if ppo_data["eval"] is not None:
        ppo_means, ppo_stds = smooth_data(ppo_data["eval"], window_size)
        x_points = np.arange(len(ppo_means)) * window_size + window_size // 2
        plt.plot(
            x_points, 
            ppo_means, 
            label="PPO Only", 
            color="black",
            linestyle='--',
            linewidth=2
        )
        plt.fill_between(
            x_points, 
            ppo_means - ppo_stds, 
            ppo_means + ppo_stds, 
            alpha=0.2, 
            color="black"
        )
    
    plt.title("PPO Fine-Tuning Evaluation Rewards (BC Pretrained)")
    plt.xlabel(f"Evaluation Episode (Window Size: {window_size})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/bc_eval_rewards.png")
    plt.close()

def create_gail_training_plot(gail_data, ppo_data):
    """Create plot for GAIL+PPO training rewards with smoothing and PPO trend"""
    plt.figure(figsize=(11, 6.5))
    data_sizes = sorted(list(gail_data.keys()))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    window_size = 50
    
    for i, data_size in enumerate(data_sizes):
        if gail_data[data_size]["training"] is not None:
            rewards = gail_data[data_size]["training"]
            means, stds = smooth_data(rewards, window_size)
            x_points = np.arange(len(means)) * window_size + window_size // 2
            color = colors[i % len(colors)]
            plt.plot(
                x_points, 
                means, 
                label=f"{data_size} expert data steps", 
                color=color,
                linestyle='-',
                linewidth=2
            )
            plt.fill_between(
                x_points, 
                means - stds, 
                means + stds, 
                alpha=0.2, 
                color=color
            )
    
    # Plot PPO trend if available
    if ppo_data["training"] is not None:
        ppo_means, ppo_stds = smooth_data(ppo_data["training"], window_size)
        x_points = np.arange(len(ppo_means)) * window_size + window_size // 2
        plt.plot(
            x_points, 
            ppo_means, 
            label="PPO Only", 
            color="black",
            linestyle='--',
            linewidth=2
        )
        plt.fill_between(
            x_points, 
            ppo_means - ppo_stds, 
            ppo_means + ppo_stds, 
            alpha=0.2, 
            color="black"
        )
    
    plt.title("GAIL+PPO Training Rewards by Expert Data Size (Averaged)")
    plt.xlabel(f"Training Steps (Window Size: {window_size})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/gail_training_rewards.png")
    plt.close()

def create_gail_eval_plot(gail_data, ppo_data):
    """Create plot for GAIL+PPO evaluation rewards with smoothing and PPO trend"""
    plt.figure(figsize=(11, 6.5))
    data_sizes = sorted(list(gail_data.keys()))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    window_size = 15  # Smaller window for eval data
    
    for i, data_size in enumerate(data_sizes):
        if gail_data[data_size]["eval"] is not None:
            rewards = gail_data[data_size]["eval"]
            means, stds = smooth_data(rewards, window_size)
            x_points = np.arange(len(means)) * window_size + window_size // 2
            color = colors[i % len(colors)]
            plt.plot(
                x_points, 
                means, 
                label=f"{data_size} expert data steps", 
                color=color,
                linestyle='-',
                linewidth=2
            )
            plt.fill_between(
                x_points, 
                means - stds, 
                means + stds, 
                alpha=0.2, 
                color=color
            )
    
    # Plot PPO trend if available
    if ppo_data["eval"] is not None:
        ppo_means, ppo_stds = smooth_data(ppo_data["eval"], window_size)
        x_points = np.arange(len(ppo_means)) * window_size + window_size // 2
        plt.plot(
            x_points, 
            ppo_means, 
            label="PPO Only", 
            color="black",
            linestyle='--',
            linewidth=2
        )
        plt.fill_between(
            x_points, 
            ppo_means - ppo_stds, 
            ppo_means + ppo_stds, 
            alpha=0.2, 
            color="black"
        )
    
    plt.title("PPO Fine-Tuning Evaluation Rewards (GAIL Pretraind)")
    plt.xlabel(f"Evaluation Episode (Window Size: {window_size})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/gail_eval_rewards.png")
    plt.close()

if __name__ == "__main__":
    create_plots()
    print(f"Plots saved to {OUTPUT_DIR}/")
