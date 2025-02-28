import pandas as pd
import matplotlib.pyplot as plt

# Load training log
log_file = "training_log.csv"
df = pd.read_csv(log_file)

# Plot avg reward vs. iteration
plt.figure(figsize=(10, 5))
plt.plot(df["iteration"], df["total_reward"], label="Reward", linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Total Reward")
plt.title("PPO Training Progress")
plt.legend()
plt.grid()

# Save the figure
plot_path = "ppo_training_curve.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # Save with high resolution
print(f"Plot saved as {plot_path}")

# Show the plot (optional)
plt.show()
