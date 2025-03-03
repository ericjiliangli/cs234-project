#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import glob
from PIL import Image

# Directory containing expert data
EXPERT_DATA_DIR = "expert_data"
EPOCHS = 2000
OUTPUT_DIR = "outputs"

def extract_data_size(filename):
    """Extract data size from filename pattern expert_data_X_reward=Y.npz"""
    match = re.search(r'expert_data_(\d+)_reward', filename)
    if match:
        return int(match.group(1))
    return None

def run_experiment(data_file, epochs):
    """Run behavior cloning with specific data file"""
    data_size = extract_data_size(data_file)
    if data_size is None:
        print(f"Could not extract data size from {data_file}, skipping...")
        return None
    
    experiment_dir = f"{OUTPUT_DIR}/size_{data_size}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Assuming your training script is called train.py
    cmd = [
        "python", "bc/train.py",
        "--expert_data", os.path.join(EXPERT_DATA_DIR, data_file),
        "--n_epoch", str(epochs)
    ]
    
    print(f"Running BC with data file: {data_file} (size: {data_size})...")
    subprocess.run(cmd, check=True)
    
    return {
        "size": data_size,
        "dir": experiment_dir,
        "file": data_file
    }

def collect_results():
    """Collect results from all experiment directories"""
    results = []
    
    # Find all experiment directories
    for exp_dir in glob.glob(f"{OUTPUT_DIR}/size_*"):
        data_size = int(exp_dir.split("_")[-1])
        
        # Check if rewards.png exists
        rewards_path = os.path.join(exp_dir, "rewards.png")
        if os.path.exists(rewards_path):
            results.append({
                "size": data_size,
                "dir": exp_dir,
                "rewards_img": rewards_path
            })
    
    # Sort by data size
    results.sort(key=lambda x: x["size"])
    return results

def create_comparison_plots(results):
    """Create comparison plots from individual reward images"""
    if not results:
        print("No results found!")
        return
    
    # Extract training and testing images from rewards.png
    for result in results:
        img = Image.open(result["rewards_img"])
        width, height = img.size
        
        # Assuming rewards.png has training on left, testing on right
        # Split the image in half
        training_img = img.crop((0, 0, width//2, height))
        testing_img = img.crop((width//2, 0, width, height))
        
        # Save the split images
        training_path = os.path.join(result["dir"], "training_rewards.png")
        testing_path = os.path.join(result["dir"], "testing_rewards.png")
        
        training_img.save(training_path)
        testing_img.save(testing_path)
        
        result["training_img"] = training_path
        result["testing_img"] = testing_path
    
    # Create a figure with all training rewards
    plt.figure(figsize=(12, 8))
    plt.title("Training Rewards by Expert Data Size")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    for result in results:
        data_size = result["size"]
        plt.plot([], [], label=f"{data_size} steps")  # Placeholder for legend
    
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/training_comparison.png", dpi=300)
    plt.close()
    
    # Create a figure with all testing rewards
    plt.figure(figsize=(12, 8))
    plt.title("Testing Rewards by Expert Data Size")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    for result in results:
        data_size = result["size"]
        plt.plot([], [], label=f"{data_size} steps")  # Placeholder for legend
    
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/testing_comparison.png", dpi=300)
    plt.close()
    
    # Create a combined figure showing all individual plots
    fig, axes = plt.subplots(2, len(results), figsize=(4*len(results), 8))
    
    for i, result in enumerate(results):
        data_size = result["size"]
        
        # Add training image
        train_img = plt.imread(result["training_img"])
        axes[0, i].imshow(train_img)
        axes[0, i].set_title(f"Training - {data_size} steps")
        axes[0, i].axis('off')
        
        # Add testing image
        test_img = plt.imread(result["testing_img"])
        axes[1, i].imshow(test_img)
        axes[1, i].set_title(f"Testing - {data_size} steps")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/all_rewards_comparison.png", dpi=300)
    plt.close()
    
    print(f"Comparison plots saved to {OUTPUT_DIR}/")

def create_plots_from_npy():
    """Create plots from saved NumPy arrays and generate comparison plots"""
    # Dictionary to store data for all experiments
    all_training_data = {}
    all_testing_data = {}
    data_sizes = []
    
    # Find all BC experiment directories
    for exp_dir in glob.glob(f"{OUTPUT_DIR}/bc_BipedalWalker-v3_*data_*epochs"):
        # Extract data size from directory name
        data_size = int(exp_dir.split("_")[-2].replace("data", ""))
        epochs = int(exp_dir.split("_")[-1].replace("epochs", ""))
        data_sizes.append(data_size)
        
        # Check if reward files exist
        training_rewards_path = os.path.join(exp_dir, "training_rewards.npy")
        test_rewards_path = os.path.join(exp_dir, "test_rewards.npy")
        
        if os.path.exists(training_rewards_path) and os.path.exists(test_rewards_path):
            # Load data
            training_rewards = np.load(training_rewards_path)
            test_rewards = np.load(test_rewards_path)
            
            # Store data for comparison plots
            all_training_data[data_size] = training_rewards
            all_testing_data[data_size] = test_rewards
            
            # Create individual figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot training rewards
            ax1.plot(training_rewards[:, 0], training_rewards[:, 1])
            ax1.set_title(f"Training Rewards ({data_size} data points)")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Reward")
            ax1.grid(True)
            
            # Plot test rewards
            ax2.plot(np.arange(len(test_rewards)), test_rewards)
            ax2.set_title(f"Testing Rewards ({data_size} data points)")
            ax2.set_xlabel("Evaluation")
            ax2.set_ylabel("Reward")
            ax2.grid(True)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, "rewards.png"), dpi=300)
            plt.close()
            
            print(f"Created plot for {data_size} data points")
    
    # Sort data sizes for consistent plotting
    data_sizes = sorted(list(set(data_sizes)))
    
    # Create a comprehensive figure for all training rewards
    plt.figure(figsize=(10, 6))
    plt.title("Impact of Expert Data Quantity on Behavior Cloning Training Performance")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_sizes)))
    
    for i, size in enumerate(data_sizes):
        if size in all_training_data:
            rewards = all_training_data[size]
            plt.plot(rewards[:, 0], rewards[:, 1], label=f"{size} steps", color=colors[i])
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_rewards_comparison.png", dpi=300)
    plt.close()
    
    # Create a comprehensive figure for all testing rewards
    plt.figure(figsize=(10, 6))
    plt.title("Impact of Expert Data Quantity on Behavior Cloning Testing Performance")
    plt.xlabel("Evaluation")
    plt.ylabel("Reward")
    
    for i, size in enumerate(data_sizes):
        if size in all_testing_data:
            rewards = all_testing_data[size]
            plt.plot(np.arange(len(rewards)), rewards, label=f"{size} steps", color=colors[i])
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/testing_rewards_comparison.png", dpi=300)
    plt.close()
    
    print(f"Comparison plots saved to {OUTPUT_DIR}/")

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all expert data files
    data_files = [f for f in os.listdir(EXPERT_DATA_DIR) if f.startswith("expert_data_") and f.endswith(".npz")]
    
    # Run experiments
    experiment_results = []
    for data_file in data_files:
        result = run_experiment(data_file, EPOCHS)
        if result:
            experiment_results.append(result)
    
    # Wait for all experiments to complete
    input("Press Enter when all experiments have completed...")
    
    # Collect and analyze results
    results = collect_results()
    create_comparison_plots(results)

if __name__ == "__main__":
    create_plots_from_npy()