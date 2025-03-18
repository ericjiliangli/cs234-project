import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

# Define directories
BC_DIR = "pretrain_PPO"
GAIL_DIR = "pretrain_PPO"  # Current directory where GAIL outputs are
PPO_DIR = "pretrain_PPO/PPO-result"  # Directory with PPO-only results
OUTPUT_DIR = "pretrain_PPO/comparison_plots"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_data_size(directory_name):
    """Extract data size from directory name"""
    # For BC directories
    if "bc_Pendulum" in directory_name:
        match = re.search(r'(\d+)data', directory_name)
        if match:
            return int(match.group(1))
    # For GAIL directories
    elif "gail_output-" in directory_name:
        match = re.search(r'gail_output-(\d+)_expert', directory_name)
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
    
    # Calculate number of windows
    num_windows = len(data) // window_size
    
    # Prepare arrays for means and stds
    means = np.zeros(num_windows)
    stds = np.zeros(num_windows)
    
    # Calculate mean and std for each window
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_data = data[start_idx:end_idx]
        means[i] = np.mean(window_data)
        stds[i] = np.std(window_data)
    
    return means, stds

def create_plots():
    """Create separate plots for BC+PPO and GAIL+PPO with PPO-only for comparison"""
    # Collect data
    bc_data = {}  # {data_size: {"training": array, "eval": array}}
    gail_data = {}
    
    # Load PPO-only data
    ppo_training_rewards, ppo_eval_rewards = load_rewards_from_npz(PPO_DIR)
    
    # Find all BC directories
    for bc_dir in glob.glob(f"{BC_DIR}/outputs-Pendulum-bc_Pendulum-v1_*data_10000epochs"):
        data_size = extract_data_size(bc_dir)
        if data_size is not None:
            training_rewards, eval_rewards = load_rewards_from_npz(bc_dir)
            bc_data[data_size] = {
                "training": training_rewards,
                "eval": eval_rewards
            }
    
    # Find all GAIL directories
    for gail_dir in glob.glob(f"{GAIL_DIR}/gail_output-*_expert-Pendulum-v1"):
        data_size = extract_data_size(gail_dir)
        if data_size is not None:
            training_rewards, eval_rewards = load_rewards_from_npz(gail_dir)
            gail_data[data_size] = {
                "training": training_rewards,
                "eval": eval_rewards
            }
    
    # Create separate plots for BC+PPO with PPO-only
    create_bc_training_plot(bc_data, ppo_training_rewards)
    create_bc_eval_plot(bc_data, ppo_eval_rewards)
    
    # Create separate plots for GAIL+PPO with PPO-only
    create_gail_training_plot(gail_data, ppo_training_rewards)
    create_gail_eval_plot(gail_data, ppo_eval_rewards)

def create_bc_training_plot(bc_data, ppo_training_rewards):
    """Create plot for BC+PPO training rewards with smoothing, including PPO-only"""
    plt.figure(figsize=(11, 6.5))
    
    # Sort data sizes for consistent colors
    data_sizes = sorted(list(bc_data.keys()))
    
    # Define distinct colors but use the same line style
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Window size for smoothing (adjust as needed)
    window_size = 50
    
    # Plot BC+PPO training rewards
    for i, data_size in enumerate(data_sizes):
        if bc_data[data_size]["training"] is not None:
            rewards = bc_data[data_size]["training"]
            
            # Smooth the data
            means, stds = smooth_data(rewards, window_size)
            
            # Create x-axis points (representing the center of each window)
            x_points = np.arange(len(means)) * window_size + window_size // 2
            
            # Plot mean line with consistent line style
            color = colors[i % len(colors)]
            
            plt.plot(
                x_points, 
                means, 
                label=f"{data_size} expert data steps", 
                color=color,
                linestyle='-',  # Use solid line for all
                linewidth=2
            )
            
            # Plot standard deviation as shaded area
            plt.fill_between(
                x_points, 
                means - stds, 
                means + stds, 
                alpha=0.2, 
                color=color
            )
    
    # Add PPO-only training rewards
    if ppo_training_rewards is not None:
        # Smooth the data
        means, stds = smooth_data(ppo_training_rewards, window_size)
        
        # Create x-axis points
        x_points = np.arange(len(means)) * window_size + window_size // 2
        
        # Plot PPO-only with black dashed line
        plt.plot(
            x_points, 
            means, 
            label="PPO only", 
            color='black',
            linestyle='--',  # Use dashed line for PPO-only
            linewidth=2
        )
        
        # Plot standard deviation as shaded area
        plt.fill_between(
            x_points, 
            means - stds, 
            means + stds, 
            alpha=0.2, 
            color='black'
        )
    
    plt.title("PPO Fine-Tuning Evaluation Rewards (BC Pretrained)")
    plt.xlabel(f"Training Steps (Window Size: {window_size})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/bc_training_rewards.png")
    plt.close()

def create_bc_eval_plot(bc_data, ppo_eval_rewards):
    """Create plot for BC+PPO evaluation rewards with smoothing, including PPO-only"""
    plt.figure(figsize=(11, 6.5))
    
    # Sort data sizes for consistent colors
    data_sizes = sorted(list(bc_data.keys()))
    
    # Define distinct colors but use the same line style
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Window size for smoothing (adjust as needed)
    window_size = 15  # Smaller window for eval data which typically has fewer points
    
    # Plot BC+PPO evaluation rewards
    for i, data_size in enumerate(data_sizes):
        if bc_data[data_size]["eval"] is not None:
            rewards = bc_data[data_size]["eval"]
            
            # Smooth the data
            means, stds = smooth_data(rewards, window_size)
            
            # Create x-axis points (representing the center of each window)
            x_points = np.arange(len(means)) * window_size + window_size // 2
            
            # Plot mean line with consistent line style
            color = colors[i % len(colors)]
            
            plt.plot(
                x_points, 
                means, 
                label=f"{data_size} expert data steps", 
                color=color,
                linestyle='-',  # Use solid line for all
                linewidth=2
            )
            
            # Plot standard deviation as shaded area
            plt.fill_between(
                x_points, 
                means - stds, 
                means + stds, 
                alpha=0.2, 
                color=color
            )
    
    # Add PPO-only evaluation rewards
    if ppo_eval_rewards is not None:
        # Smooth the data
        means, stds = smooth_data(ppo_eval_rewards, window_size)
        
        # Create x-axis points
        x_points = np.arange(len(means)) * window_size + window_size // 2
        
        # Plot PPO-only with black dashed line
        plt.plot(
            x_points, 
            means, 
            label="PPO only", 
            color='black',
            linestyle='--',  # Use dashed line for PPO-only
            linewidth=2
        )
        
        # Plot standard deviation as shaded area
        plt.fill_between(
            x_points, 
            means - stds, 
            means + stds, 
            alpha=0.2, 
            color='black'
        )
    
    plt.title("PPO Fine-Tuning Evaluation Rewards (BC Pretrained)")
    plt.xlabel(f"Evaluation Episode (Window Size: {window_size})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/bc_eval_rewards.png")
    plt.close()

def create_gail_training_plot(gail_data, ppo_training_rewards):
    """Create plot for GAIL+PPO training rewards with smoothing, including PPO-only"""
    plt.figure(figsize=(11, 6.5))
    
    # Sort data sizes for consistent colors
    data_sizes = sorted(list(gail_data.keys()))
    
    # Define distinct colors but use the same line style
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Window size for smoothing (adjust as needed)
    window_size = 50
    
    # Plot GAIL+PPO training rewards
    for i, data_size in enumerate(data_sizes):
        if gail_data[data_size]["training"] is not None:
            rewards = gail_data[data_size]["training"]
            
            # Smooth the data
            means, stds = smooth_data(rewards, window_size)
            
            # Create x-axis points (representing the center of each window)
            x_points = np.arange(len(means)) * window_size + window_size // 2
            
            # Plot mean line with consistent line style
            color = colors[i % len(colors)]
            
            plt.plot(
                x_points, 
                means, 
                label=f"{data_size} expert data steps", 
                color=color,
                linestyle='-',  # Use solid line for all
                linewidth=2
            )
            
            # Plot standard deviation as shaded area
            plt.fill_between(
                x_points, 
                means - stds, 
                means + stds, 
                alpha=0.2, 
                color=color
            )
    
    # Add PPO-only training rewards
    if ppo_training_rewards is not None:
        # Smooth the data
        means, stds = smooth_data(ppo_training_rewards, window_size)
        
        # Create x-axis points
        x_points = np.arange(len(means)) * window_size + window_size // 2
        
        # Plot PPO-only with black dashed line
        plt.plot(
            x_points, 
            means, 
            label="PPO only", 
            color='black',
            linestyle='--',  # Use dashed line for PPO-only
            linewidth=2
        )
        
        # Plot standard deviation as shaded area
        plt.fill_between(
            x_points, 
            means - stds, 
            means + stds, 
            alpha=0.2, 
            color='black'
        )
    
    plt.title("PPO Fine-Tuning Evaluation Rewards (GAIL Pretrained)")
    plt.xlabel(f"Training Steps (Window Size: {window_size})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/gail_training_rewards.png")
    plt.close()

def create_gail_eval_plot(gail_data, ppo_eval_rewards):
    """Create plot for GAIL+PPO evaluation rewards with smoothing, including PPO-only"""
    plt.figure(figsize=(11, 6.5))
    
    # Sort data sizes for consistent colors
    data_sizes = sorted(list(gail_data.keys()))
    
    # Define distinct colors but use the same line style
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Window size for smoothing (adjust as needed)
    window_size = 15  # Smaller window for eval data which typically has fewer points
    
    # Plot GAIL+PPO evaluation rewards
    for i, data_size in enumerate(data_sizes):
        if gail_data[data_size]["eval"] is not None:
            rewards = gail_data[data_size]["eval"]
            
            # Smooth the data
            means, stds = smooth_data(rewards, window_size)
            
            # Create x-axis points (representing the center of each window)
            x_points = np.arange(len(means)) * window_size + window_size // 2
            
            # Plot mean line with consistent line style
            color = colors[i % len(colors)]
            
            plt.plot(
                x_points, 
                means, 
                label=f"{data_size} expert data steps", 
                color=color,
                linestyle='-',  # Use solid line for all
                linewidth=2
            )
            
            # Plot standard deviation as shaded area
            plt.fill_between(
                x_points, 
                means - stds, 
                means + stds, 
                alpha=0.2, 
                color=color
            )
    
    # Add PPO-only evaluation rewards
    if ppo_eval_rewards is not None:
        # Smooth the data
        means, stds = smooth_data(ppo_eval_rewards, window_size)
        
        # Create x-axis points
        x_points = np.arange(len(means)) * window_size + window_size // 2
        
        # Plot PPO-only with black dashed line
        plt.plot(
            x_points, 
            means, 
            label="PPO only", 
            color='black',
            linestyle='--',  # Use dashed line for PPO-only
            linewidth=2
        )
        
        # Plot standard deviation as shaded area
        plt.fill_between(
            x_points, 
            means - stds, 
            means + stds, 
            alpha=0.2, 
            color='black'
        )
    
    plt.title("PPO Fine-Tuning Evaluation Rewards (GAIL Pretrained)")
    plt.xlabel(f"Evaluation Episode (Window Size: {window_size})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/gail_eval_rewards.png")
    plt.close()

if __name__ == "__main__":
    create_plots()
    print(f"Plots saved to {OUTPUT_DIR}/")