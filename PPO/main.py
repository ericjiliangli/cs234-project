import gym
import torch
import numpy as np
import os
from model import create_networks
from ppo import PPO, Memory
import pandas as pd
from tqdm import tqdm

class PartialObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, mask):
        super().__init__(env)
        self.mask = mask  # A binary mask with the same shape as the observation

    def observation(self, obs):
        return obs * self.mask
    

# Hyperparameters
ENV_NAME = "BipedalWalker-v3"
SEED = 42
LR = 3e-4
GAMMA = 0.99
LAM = 0.95
EPS_CLIP = 0.2
EPOCHS = 5
BATCH_SIZE = 64
MAX_EPISODES = 5000
MAX_STEPS = 1600  # Limit steps per episode
SAVE_INTERVAL = 100  # Save model every N episodes
CHECKPOINT_DIR = "checkpoints"

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Create environment and apply masking
env = gym.make("BipedalWalker-v3")
obs_dim = env.observation_space.shape[0]
mask = np.ones(obs_dim)  # Full observability initially
mask[14:] = 0  # Masking out LIDAR features (indices 14 to 23)
env = PartialObsWrapper(env, mask)

state_dim = obs_dim  # State dim remains the same after masking
action_dim = env.action_space.shape[0] if env.action_space.shape else env.action_space.n
discrete = isinstance(env.action_space, gym.spaces.Discrete)

# Initialize networks and PPO agent
policy_net, value_net = create_networks(state_dim, action_dim, discrete)
ppo_agent = PPO(policy_net, value_net, lr=LR, gamma=GAMMA, lam=LAM, eps_clip=EPS_CLIP, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Load checkpoint if available
checkpoint_path = os.path.join(CHECKPOINT_DIR, "ppo_checkpoint.pth")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    policy_net.load_state_dict(checkpoint["policy_net"])
    value_net.load_state_dict(checkpoint["value_net"])
    print(f"Loaded checkpoint from {checkpoint_path}")

# Training loop
memory = Memory()
episode_rewards = []
episode_lengths = []
progress_bar = tqdm(range(1, MAX_EPISODES + 1), desc="Training PPO", unit="episode")


LOG_FILE = "training_log.csv"

# Initialize logging
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["iteration", "avg_reward"]).to_csv(LOG_FILE, index=False)

print(f"Training {ENV_NAME} with partial observability...")

for episode in progress_bar:
    state = env.reset()[0]  # Gym API returns (state, info) tuple
    total_reward = 0

    steps = 0
    for step in range(MAX_STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        distb = policy_net(state_tensor)
        action = distb.sample()
        log_prob = distb.log_prob(action).sum()  # Sum in case of multiple action dimensions

        action_np = action.detach().numpy()
        next_state, reward, done, _, _ = env.step(action_np)

        value = value_net(state_tensor).detach().item()
        memory.store(state, action_np, log_prob.item(), reward, done, value)

        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    
    # Update PPO after collecting experience
    ppo_agent.update(memory)

    # Logging
    avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths[-10:]) if len(episode_lengths) >= 10 else np.mean(episode_lengths)
    print(f"Episode {episode}/{MAX_EPISODES}, Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Length: {step+1}, Avg Length: {avg_length:.2f}")

    # progress_bar.set_postfix({"Last Reward": total_reward,"Avg Reward": avg_reward, "Avg Length": avg_length})

    
    df = pd.read_csv(LOG_FILE)
    new_entry = pd.DataFrame({"iteration": [episode], "reward": [total_reward]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)
    
    # Save model checkpoints
    if episode % SAVE_INTERVAL == 0:
        print("\nSaving the model ... ", end="")
        checkpoint = {
            "it": episode,
            "policy_net": policy_net.state_dict(),
            "value_net": value_net.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

env.close()
