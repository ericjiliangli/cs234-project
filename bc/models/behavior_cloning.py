# behavior_cloning.py
import numpy as np
import gym
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.policy_network import PolicyNetwork
from utils.utils import to_input, ExpertDataset
from matplotlib import pyplot as plt
import argparse
import os

class BehavioralCloning:
    def __init__(self, env_name, expert_data_path, num_trajectories, n_epoch,
                 batch_size, lr, n, compare, eval_frequency=5,
                 num_test_episodes=10, max_test_steps=500, test_seeds=None):

        self.env_name = env_name
        self.expert_data_path = expert_data_path
        self.num_trajectories = num_trajectories
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.n = n  # Window size
        self.compare = compare
        self.eval_frequency = eval_frequency  # Added evaluation frequency parameter
        self.num_test_episodes = num_test_episodes
        self.max_test_steps = max_test_steps
        self.test_seeds = test_seeds

        self.env = gym.make(self.env_name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

        # Instantiate the policy network
        self.policy = PolicyNetwork(self.state_dim * self.n, self.action_dim, discrete=False)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.dataloader = None

    def load_expert_data(self):
        with np.load(self.expert_data_path) as data:
            obs = torch.tensor(data['obs'], dtype=torch.float32)  # Convert to tensor
            acts = torch.tensor(data['acts'], dtype=torch.float32)  # Convert to tensor

        # Reshape data to include the episode dimension if necessary
        obs = obs.view(-1, 1, obs.shape[-1])
        acts = acts.view(-1, 1, acts.shape[-1])

        states, actions = to_input(obs, acts, self.n, self.compare)
        self.dataloader = DataLoader(ExpertDataset(states, actions), batch_size=self.batch_size, shuffle=True)

    def train(self):
        self.load_expert_data()
        loss_history = []
        training_rewards = []

        for epoch in range(self.n_epoch):
            # Training phase - update the model
            total_loss = 0
            for batch_states, batch_actions in self.dataloader:
                self.optimizer.zero_grad()
                action_distribution = self.policy(batch_states)
                predicted_actions = action_distribution.mean
                loss = self.criterion(predicted_actions, batch_actions)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.dataloader)
            loss_history.append(avg_loss)
            
            # Only evaluate at specified frequency or at first/last epoch
            if (epoch + 1) % self.eval_frequency == 0 or epoch == 0 or epoch == self.n_epoch - 1:
                # Use the evaluate method with a smaller number of episodes
                self.num_test_episodes_backup = self.num_test_episodes  
                self.num_test_episodes = 3  
                
                # Call the evaluate method
                epoch_rewards = self.evaluate(verbose=False) 
                
                # Restore original value
                self.num_test_episodes = self.num_test_episodes_backup
                
                avg_reward = sum(epoch_rewards) / len(epoch_rewards)
                training_rewards.append((epoch, avg_reward))  
                
                print(f"Epoch {epoch + 1}/{self.n_epoch}, Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")
            else:
                # For non-evaluation epochs, just print the loss
                print(f"Epoch {epoch + 1}/{self.n_epoch}, Loss: {avg_loss:.4f}")

        # Save only the evaluation results
        np.save("training_rewards.npy", np.array(training_rewards))
        return loss_history, training_rewards

    def save_model(self, path):
        """Save the model to the specified path."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")

    def evaluate(self, verbose=True):
        self.policy.eval()
        test_rewards = []

        for seed in self.test_seeds or [np.random.randint(0, 10000) for _ in range(self.num_test_episodes)]:
            try:
                # Try newer Gym API
                state, _ = self.env.reset(seed=seed)
            except (TypeError, ValueError):
                # Fall back to older Gym API
                state = self.env.reset()
                if hasattr(self.env, 'seed'):
                    self.env.seed(seed)
            
            # Convert state to numpy array if it's a list
            if isinstance(state, list):
                state = np.array(state, dtype=np.float32)
            
            # Skip this episode if state is invalid
            if not isinstance(state, np.ndarray) or state.size == 0:
                print(f"Warning: Invalid state received for seed {seed}. Skipping this episode.")
                continue
            
            episode_reward = 0
            step = 0

            while step < self.max_test_steps:
                # Ensure state is properly shaped for the network
                if len(state.shape) == 1:
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                else:
                    state_tensor = torch.from_numpy(state).float()
                
                with torch.no_grad():
                    action_distribution = self.policy(state_tensor)
                    action = action_distribution.mean.cpu().numpy()[0]
                
                # Clip action to environment's range
                action = np.clip(action, -1, 1)
                
                try:
                    # Try newer Gym API
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                except ValueError:
                    # Fall back to older Gym API
                    next_state, reward, done, _ = self.env.step(action)
                
                episode_reward += reward
                
                # Convert next_state to numpy array if it's a list
                if isinstance(next_state, list):
                    next_state = np.array(next_state, dtype=np.float32)
                
                state = next_state
                step += 1
                
                if done:
                    break

            test_rewards.append(episode_reward)
            if verbose:
                print(f"Seed: {seed}, Total Reward: {episode_reward:.2f}")

        self.env.close()
        return test_rewards

    def load_model(self, load_path):
        self.policy.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")