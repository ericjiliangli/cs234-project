import torch
import gym
import numpy as np
from model import PolicyNetwork

def evaluate_policy(env_name="BipedalWalker-v3", checkpoint_path="checkpoints/ppo_checkpoint.pth", episodes=10, render=False):
    """
    Loads the trained policy network from the latest checkpoint and evaluates it on the environment.

    Args:
        env_name (str): The name of the Gym environment.
        checkpoint_path (str): Path to the saved checkpoint file.
        episodes (int): Number of evaluation episodes.
        render (bool): Whether to render the environment.

    Returns:
        avg_reward (float): The average reward over the evaluation episodes.
        avg_length (float): The average episode length over evaluation episodes.
    """

    # Load environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if env.action_space.shape else env.action_space.n
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Load policy network
    policy_net = PolicyNetwork(state_dim, action_dim, discrete)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))  # Load checkpoint on CPU
    policy_net.load_state_dict(checkpoint["policy_net"])
    policy_net.eval()  # Set to evaluation mode

    total_rewards = []
    total_lengths = []

    for episode in range(episodes):
        state = env.reset()[0]  # Reset environment
        total_reward = 0
        steps = 0

        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                distb = policy_net(state_tensor)
                action = distb.sample()

            action_np = action.numpy()
            state, reward, done, _, _ = env.step(action_np)

            if render:
                env.render()

            total_reward += reward
            steps += 1

            if done:
                break

        total_rewards.append(total_reward)
        total_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Length = {steps}")

    env.close()

    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(total_lengths)
    print(f"\nEvaluation Complete: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length}")

    return avg_reward, avg_length

if __name__ == "__main__":
    evaluate_policy()
