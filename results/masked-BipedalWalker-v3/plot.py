import os
import numpy as np
import matplotlib.pyplot as plt

# Define directories (assumes you are in the base directory)
BC_DIR = "outputs-bipedalWalker-bc_BipedalWalker-v3_5000data_10000epochs"
GAIL_DIR = "gail-masked-BipedalWalker-v3"
PPO_DIR = "PPO"
OUTPUT_DIR = "result"

# Output filenames for training and evaluation plots
TRAIN_OUTPUT_FILENAME = "BC+PPO_vs_GAIL+PPO_on_Masked_BipedalWalker-v3_training.png"
EVAL_OUTPUT_FILENAME = "BC+PPO_vs_GAIL+PPO_on_Masked_BipedalWalker-v3_eval.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_rewards_from_npz(directory):
    """
    Load training and evaluation rewards from NPZ files in the given directory.
    Expects files "train_reward_masked.np.npz" and "eval_reward_masked.np.npz".
    """
    train_file = os.path.join(directory, "train_reward_masked.np.npz")
    eval_file = os.path.join(directory, "eval_reward_masked.np.npz")
    
    train_rewards = None
    eval_rewards = None

    if os.path.exists(train_file):
        data = np.load(train_file)
        # Adjust the key if necessary; we assume the key is "rewards"
        if "rewards" in data:
            train_rewards = data["rewards"]
    if os.path.exists(eval_file):
        data = np.load(eval_file)
        if "rewards" in data:
            eval_rewards = data["rewards"]
    
    return train_rewards, eval_rewards

def smooth_data(data, window_size):
    """
    Smooth data using a sliding window.
    Returns a tuple of (means, stds). If the data length is less than or equal to the window,
    return a single window value.
    """
    if data is None or len(data) == 0:
        return None, None
    if len(data) <= window_size:
        return np.array([np.mean(data)]), np.array([np.std(data)])
    
    num_windows = len(data) // window_size
    means = np.array([np.mean(data[i*window_size:(i+1)*window_size]) for i in range(num_windows)])
    stds = np.array([np.std(data[i*window_size:(i+1)*window_size]) for i in range(num_windows)])
    return means, stds

def plot_training_rewards(bc_train, gail_train, ppo_train, window_size=50):
    """
    Create and save the training rewards plot.
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))
    
    # Plot BC+PPO training rewards (blue)
    if bc_train is not None:
        bc_means, bc_stds = smooth_data(bc_train, window_size)
        if bc_means is not None:
            x = np.arange(len(bc_means)) * window_size + window_size // 2
            ax.plot(x, bc_means, label="BC+PPO", color="blue", linestyle="-", linewidth=2)
            ax.fill_between(x, bc_means - bc_stds, bc_means + bc_stds, color="blue", alpha=0.2)
    
    # Plot GAIL+PPO training rewards (red)
    if gail_train is not None:
        gail_means, gail_stds = smooth_data(gail_train, window_size)
        if gail_means is not None:
            x = np.arange(len(gail_means)) * window_size + window_size // 2
            ax.plot(x, gail_means, label="GAIL+PPO", color="red", linestyle="-", linewidth=2)
            ax.fill_between(x, gail_means - gail_stds, gail_means + gail_stds, color="red", alpha=0.2)
    
    # Plot PPO trend (dotted black)
    if ppo_train is not None:
        ppo_means, ppo_stds = smooth_data(ppo_train, window_size)
        if ppo_means is not None:
            x = np.arange(len(ppo_means)) * window_size + window_size // 2
            ax.plot(x, ppo_means, label="PPO Only", color="black", linestyle='--', linewidth=2)
            ax.fill_between(x, ppo_means - ppo_stds, ppo_means + ppo_stds, color="black", alpha=0.2)
    
    ax.set_title("Training Rewards: BC+PPO vs. GAIL+PPO on Masked BipedalWalker-v3")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Reward")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.grid(True)
    output_path = os.path.join(OUTPUT_DIR, TRAIN_OUTPUT_FILENAME)
    plt.savefig(output_path)
    plt.close()
    print(f"Training rewards plot saved to {output_path}")

def plot_eval_rewards(bc_eval, gail_eval, ppo_eval, window_size=10):
    """
    Create and save the evaluation rewards plot.
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))
    
    # Plot BC+PPO evaluation rewards (blue)
    if bc_eval is not None:
        bc_means, bc_stds = smooth_data(bc_eval, window_size)
        if bc_means is not None:
            x = np.arange(len(bc_means)) * window_size + window_size // 2
            ax.plot(x, bc_means, label="BC+PPO 5000 expert data steps", color="blue", linestyle="-", linewidth=2)
            ax.fill_between(x, bc_means - bc_stds, bc_means + bc_stds, color="blue", alpha=0.2)
    
    # Plot GAIL+PPO evaluation rewards (red)
    if gail_eval is not None:
        gail_means, gail_stds = smooth_data(gail_eval, window_size)
        if gail_means is not None:
            x = np.arange(len(gail_means)) * window_size + window_size // 2
            ax.plot(x, gail_means, label="GAIL+PPO 5000 expert data steps", color="red", linestyle="-", linewidth=2)
            ax.fill_between(x, gail_means - gail_stds, gail_means + gail_stds, color="red", alpha=0.2)
    
    # Plot PPO trend (dotted black)
    if ppo_eval is not None:
        ppo_means, ppo_stds = smooth_data(ppo_eval, window_size)
        if ppo_means is not None:
            x = np.arange(len(ppo_means)) * window_size + window_size // 2
            ax.plot(x, ppo_means, label="PPO Only", color="black", linestyle='--', linewidth=2)
            ax.fill_between(x, ppo_means - ppo_stds, ppo_means + ppo_stds, color="black", alpha=0.2)
    
    ax.set_title("PPO Fine-Tuning Evaluation Rewards (BC & GAIL Pretrained)")
    ax.set_xlabel(f"Evaluation Episode (Window Size: {window_size})")
    ax.set_ylabel("Reward")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.grid(True)
    output_path = os.path.join(OUTPUT_DIR, EVAL_OUTPUT_FILENAME)
    plt.savefig(output_path)
    plt.close()
    print(f"Evaluation rewards plot saved to {output_path}")

def main():
    # Load training and evaluation rewards from each folder
    bc_train, bc_eval = load_rewards_from_npz(BC_DIR)
    gail_train, gail_eval = load_rewards_from_npz(GAIL_DIR)
    ppo_train, ppo_eval = load_rewards_from_npz(PPO_DIR)
    
    # Plot training rewards (using a smoothing window of 50 steps)
    plot_training_rewards(bc_train, gail_train, ppo_train, window_size=50)
    
    # Plot evaluation rewards (using a smoothing window of 10 episodes)
    plot_eval_rewards(bc_eval, gail_eval, ppo_eval, window_size=10)

if __name__ == "__main__":
    main()
