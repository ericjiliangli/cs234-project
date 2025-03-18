import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

# Define directories
BC_PENDULUM_DIR = "outputs/Pendulum"  # Directory with Pendulum BC results
BC_BIPEDAL_DIR = "outputs/bipedalWalker"   # Directory with BipedalWalker BC results
OUTPUT_DIR = "bc_comparison_plots"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_data_size(directory_name):
    """Extract data size from directory name"""
    # For BC Pendulum directories
    if "bc_Pendulum" in directory_name:
        match = re.search(r'(\d+)data', directory_name)
        if match:
            return int(match.group(1))
    # For BC BipedalWalker directories
    elif "bc_BipedalWalker" in directory_name:
        match = re.search(r'(\d+)data', directory_name)
        if match:
            return int(match.group(1))
    return None

def load_rewards_from_npy(directory):
    """Load training and evaluation rewards from NPY files in a directory"""
    training_rewards_path = os.path.join(directory, "training_rewards.npy")
    eval_rewards_path = os.path.join(directory, "test_rewards.npy")
    
    training_rewards = None
    eval_rewards = None
    
    if os.path.exists(training_rewards_path):
        try:
            training_data = np.load(training_rewards_path, allow_pickle=True)
            # Check if training data is in (epoch, reward) format
            if isinstance(training_data[0], tuple) or (isinstance(training_data, np.ndarray) and training_data.ndim == 2 and training_data.shape[1] == 2):
                # Extract just the rewards (second column)
                if isinstance(training_data[0], tuple):
                    # If it's a list of tuples
                    _, training_rewards = zip(*training_data)
                    training_rewards = np.array(training_rewards)
                else:
                    # If it's a 2D array
                    training_rewards = training_data[:, 1]
            else:
                training_rewards = training_data
            print(f"Loaded training rewards from {training_rewards_path}, shape: {training_rewards.shape if hasattr(training_rewards, 'shape') else 'unknown'}")
        except Exception as e:
            print(f"Error loading training rewards: {e}")
    else:
        print(f"Training rewards file not found: {training_rewards_path}")
    
    if os.path.exists(eval_rewards_path):
        try:
            eval_rewards = np.load(eval_rewards_path)
            print(f"Loaded evaluation rewards from {eval_rewards_path}, shape: {eval_rewards.shape if hasattr(eval_rewards, 'shape') else 'unknown'}")
        except Exception as e:
            print(f"Error loading evaluation rewards: {e}")
    else:
        print(f"Evaluation rewards file not found: {eval_rewards_path}")
    
    return training_rewards, eval_rewards

def smooth_data(data, window_size=10):
    """Calculate average and standard deviation over windows of data"""
    if data is None or len(data) == 0:
        return np.array([]), np.array([])
    
    # Convert to numpy array if it's not already
    data = np.array(data)
    
    if len(data) <= window_size:
        return np.array([np.mean(data)]), np.array([np.std(data) * 0.5])
    
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
        stds[i] = np.std(window_data) * 0.5  # Reduce std to make bands narrower
    
    return means, stds

def create_plots():
    """Create separate plots for Pendulum BC and BipedalWalker BC"""
    # Collect data
    pendulum_bc_data = {}  # {data_size: {"training": array, "eval": array}}
    bipedal_bc_data = {}
    
    # Find all Pendulum BC directories
    pendulum_dirs = glob.glob(f"{BC_PENDULUM_DIR}/bc_Pendulum-v1_*data_*epochs")
    print(f"Found {len(pendulum_dirs)} Pendulum BC directories: {pendulum_dirs}")
    
    for bc_dir in pendulum_dirs:
        data_size = extract_data_size(bc_dir)
        print(f"Extracted data size from {bc_dir}: {data_size}")
        if data_size is not None:
            training_rewards, eval_rewards = load_rewards_from_npy(bc_dir)
            pendulum_bc_data[data_size] = {
                "training": training_rewards,
                "eval": eval_rewards
            }
    
    # Find all BipedalWalker BC directories
    bipedal_dirs = glob.glob(f"{BC_BIPEDAL_DIR}/bc_BipedalWalker-v3_*data_*epochs")
    print(f"Found {len(bipedal_dirs)} BipedalWalker BC directories: {bipedal_dirs}")
    
    for bc_dir in bipedal_dirs:
        data_size = extract_data_size(bc_dir)
        print(f"Extracted data size from {bc_dir}: {data_size}")
        if data_size is not None:
            training_rewards, eval_rewards = load_rewards_from_npy(bc_dir)
            bipedal_bc_data[data_size] = {
                "training": training_rewards,
                "eval": eval_rewards
            }
    
    print(f"Pendulum BC data sizes: {list(pendulum_bc_data.keys())}")
    print(f"BipedalWalker BC data sizes: {list(bipedal_bc_data.keys())}")
    
    # Create plots for Pendulum BC
    create_pendulum_training_plot(pendulum_bc_data)
    create_pendulum_eval_plot(pendulum_bc_data)
    
    # Create plots for BipedalWalker BC
    create_bipedal_training_plot(bipedal_bc_data)
    create_bipedal_eval_plot(bipedal_bc_data)

def create_pendulum_training_plot(bc_data):
    """Create plot for Pendulum BC training rewards with smoothing"""
    plt.figure(figsize=(11, 6.5))
    
    # Sort data sizes for consistent colors
    data_sizes = sorted(list(bc_data.keys()))
    
    # Define distinct colors but use the same line style
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Window size for smoothing - adjust based on total epochs (10000)
    window_size = 10
    
    # Maximum x-axis value - use the same for both plots
    max_steps = 2000
    
    # Plot BC training rewards
    for i, data_size in enumerate(data_sizes):
        if bc_data[data_size]["training"] is not None:
            rewards = bc_data[data_size]["training"]
            
            # Smooth the data
            means, stds = smooth_data(rewards, window_size)
            
            # Create x-axis points scaled to match BipedalWalker's range
            # We'll only show the first 2000 steps (or equivalent) for Pendulum
            steps_per_point = 10000 / len(means)  # How many actual steps each point represents
            
            # Calculate how many points we need to show max_steps
            points_to_show = int(max_steps / steps_per_point)
            if points_to_show > len(means):
                points_to_show = len(means)
            
            # Use only the first points_to_show points
            means = means[:points_to_show]
            stds = stds[:points_to_show]
            
            # Create x-axis points
            x_points = np.linspace(0, max_steps, points_to_show)
            
            # Plot mean line with consistent line style
            color = colors[i % len(colors)]
            
            plt.plot(
                x_points, 
                means, 
                label=f"{data_size} expert data steps", 
                color=color,
                linestyle='-',
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
    
    plt.title("Pendulum BC Pretraining Rewards")
    plt.xlabel(f"Training Steps (Window Size: {window_size})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/pendulum_bc_training_rewards.png")
    plt.close()

def create_pendulum_eval_plot(bc_data):
    """Create plot for Pendulum BC evaluation rewards with smoothing"""
    plt.figure(figsize=(11, 6.5))
    
    # Sort data sizes for consistent colors
    data_sizes = sorted(list(bc_data.keys()))
    
    # Define distinct colors but use the same line style
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Window size for smoothing (adjust as needed)
    window_size = 10  # Smaller window for eval data which typically has fewer points
    
    # Plot BC evaluation rewards
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
    
    plt.title("Pendulum BC Evaluation Rewards")
    plt.xlabel(f"Evaluation Episode (Window Size: {window_size})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/pendulum_bc_eval_rewards.png")
    plt.close()

def create_bipedal_training_plot(bc_data):
    """Create plot for BipedalWalker BC training rewards with smoothing"""
    plt.figure(figsize=(11, 6.5))
    
    # Sort data sizes for consistent colors
    data_sizes = sorted(list(bc_data.keys()))
    
    # Define distinct colors but use the same line style
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Window size for smoothing
    window_size = 10
    
    # Maximum x-axis value - use the same for both plots
    max_steps = 2000
    
    # Plot BC training rewards
    for i, data_size in enumerate(data_sizes):
        if bc_data[data_size]["training"] is not None:
            rewards = bc_data[data_size]["training"]
            
            # Smooth the data
            means, stds = smooth_data(rewards, window_size)
            
            # Create x-axis points
            x_points = np.linspace(0, max_steps, len(means))
            
            # Plot mean line with consistent line style
            color = colors[i % len(colors)]
            
            plt.plot(
                x_points, 
                means, 
                label=f"{data_size} expert data steps", 
                color=color,
                linestyle='-',
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
    
    plt.title("BipedalWalker BC Pretraining Rewards")
    plt.xlabel(f"Training Steps (Window Size: {window_size})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/bipedal_bc_training_rewards.png")
    plt.close()

def create_bipedal_eval_plot(bc_data):
    """Create plot for BipedalWalker BC evaluation rewards with smoothing"""
    plt.figure(figsize=(11, 6.5))
    
    # Sort data sizes for consistent colors
    data_sizes = sorted(list(bc_data.keys()))
    
    # Define distinct colors but use the same line style
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Window size for smoothing (adjust as needed)
    window_size = 15  # Smaller window for eval data which typically has fewer points
    
    # Plot BC evaluation rewards
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
    
    plt.title("BipedalWalker BC Evaluation Rewards (Averaged)")
    plt.xlabel(f"Evaluation Episode (Window Size: {window_size})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/bipedal_bc_eval_rewards.png")
    plt.close()

if __name__ == "__main__":
    create_plots()
    print(f"Plots saved to {OUTPUT_DIR}/")
