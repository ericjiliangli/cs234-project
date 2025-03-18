import numpy as np
import matplotlib.pyplot as plt

# List of training and evaluation file names
train_file_names = [
    # "mask3/train_reward_masked.np.npz",
    # "mask12-13-19-23/train_reward_masked.np.npz",
    # "mask12-13-21-23/train_reward_masked.np.npz",
    # "mask14-24/train_reward_masked.np.npz",
    # "mask19-24/train_reward_masked.np.npz",
    "mask12-24/train_reward_masked.np.npz"
]

eval_file_names = [
    # "mask3/eval_reward_masked.np.npz",
    # "mask12-13-19-23/eval_reward_masked.np.npz",
    # "mask12-13-21-23/eval_reward_masked.np.npz",
    # "mask14-24/eval_reward_masked.np.npz",
    # "mask19-24/eval_reward_masked.np.npz",
    "mask12-24/eval_reward_masked.np.npz"
]

# Labels for each method
# labels = ["f3","f12-13,19-24","f12-13,21-24","f14-24", "f19-24"]
labels = ["f12-24"]

# labels = ["f3-12-13", "f12-13,19-24", "f14-24", "f19-24", "f all"]

# Function to plot and save rewards
def plot_rewards(file_names, labels, title, save_name, alpha=1.0, eval=False):
    plt.figure(figsize=(10, 6))
    
    for file_name, label in zip(file_names, labels):
        data = np.load(file_name)
        print(data)
        rewards = data['rewards']  # Assuming rewards are stored under 'arr_0'
        plt.plot(rewards, label=label,alpha=alpha)
    if eval:
        plt.axhline(y=250, color='red', linestyle='--', label="Reference: 250")
        plt.axhline(y=200, color='purple', linestyle='--', label="Reference: 200")

    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(save_name, dpi=300)
    plt.close()

# Plot and save training rewards
plot_rewards(train_file_names, labels, "Training Reward Comparison", "training_reward_12-24.png", alpha = 0.5)

# Plot and save evaluation rewards
plot_rewards(eval_file_names, labels, "Evaluation Reward Comparison", "evaluation_reward_12-24.png", eval = True)

print("Plots saved as 'training_reward.png' and 'evaluation_reward.png'.")
