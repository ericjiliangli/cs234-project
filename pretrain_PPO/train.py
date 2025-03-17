import argparse
import os
import gym
import torch
import torch.nn as nn
import numpy as np
from net import PolicyNet, ValueNet
from envSetter import EnvSetter
from ppo import PPO
from stable_baselines3.common.callbacks import EvalCallback
from utils import PartialObsWrapper
import matplotlib.pyplot as plt

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

def evaluate(env, policy_net, num_episodes=5):
    """Runs the agent for num_episodes and returns the mean reward."""
    total_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()[0][0]
        done = False
        episode_reward = 0
        
        truncated = False

        # while not done and not truncated:
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=DEVICE)
                action, _ = policy_net(state_tensor)  # Assuming policy_net.forward() returns (action, log_prob)
            action = action.cpu().numpy().flatten()
            state, reward, done, truncated, info = env.step(action.tolist())
            episode_reward += reward

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)

def main(args):
    env_name = args.env
    max_episode = args.max_episode
    ckpt = args.ckpt
    
    train_env = gym.make(env_name)
    eval_env = gym.make(env_name)
    state_dim = train_env.observation_space.shape[0]
    mask = np.ones(state_dim)
    if args.mask != "false":
        print("""Apply feature masking: 'true' (removes LIDAR features) or 'false' (full observation).""")
        # 19
        if args.mask == "19":
            print("Masking 20-24")
            mask[19:] = 0
        elif args.mask == "3":
            print("Masking 3")
            mask_indices = [3,12,13]
            mask[mask_indices] = 0
        elif args.mask == "14":
            print("Masking 14")
            mask[14:] = 0
        elif args.mask == "12":
            print("Masking 12,13,19-23")
            mask_indices = [12,13,19,20,21,22,23]
            mask[mask_indices] = 0 
        elif args.mask == "21":
            print("Masking 21")
            mask_indices = [12,13,21,22,23]
            mask[mask_indices] = 0  
        elif args.mask == "12-24":
            print("Masking 12,24")
            mask[12:] = 0
        
        train_env = PartialObsWrapper(train_env, mask)
        eval_env = PartialObsWrapper(eval_env, mask)
    if env_name in ["CartPole-v1"]:
        discrete = True
        action_dim = train_env.action_space.n
    else:
        discrete = False
        action_dim = train_env.action_space.shape[0]
    print(f"Environment {train_env} with {state_dim} states and {action_dim} actions")
    
    policy_net = PolicyNet(state_dim, action_dim, discrete)
    
    if ckpt != "none":
        policy_checkpoint = torch.load(os.path.join(ckpt, "policy.ckpt"), map_location=torch.device("cpu"))  # Load checkpoint on CPU
        policy_net.load_state_dict(policy_checkpoint)
    
    
    value_net = ValueNet(state_dim)
    
    if ckpt != "none":
        value_checkpoint = torch.load(os.path.join(ckpt, "value.ckpt"), map_location=torch.device("cpu"))  # Load checkpoint on CPU
        value_net.load_state_dict(value_checkpoint)
    
    print(f"Policy Net: {policy_net} \nValue Net: {value_net}")
    
    # detect whether I can use apple GPU mps to speed up the training
    # policy_net.to(DEVICE)
    value_net.to(DEVICE)
    print(f"Move value_net and policy_net to device: {DEVICE}")
    
    
    env_setter = EnvSetter(state_dim, action_dim, device=DEVICE)
    ppo_agent = PPO(policy_net, value_net, device=DEVICE)
    train(train_env, env_setter, policy_net, value_net, ppo_agent,max_episode, eval_env)
    train_env.close()
    
def train(env, env_setter, policy_net, value_net, agent, max_episode, eval_env, eval_freq=100):
    mean_total_reward = 0
    mean_length = 0
    
    
    save_dir = "-".join(args.ckpt.split("/")[-3:]) if args.ckpt != "none" else "PPO-result"
    if args.mask == "false":
        print("No mask applied")
        save_dir = f"{save_dir}/non-mask"
    if args.mask == "19":
        save_dir = f"{save_dir}/mask19-24"
    if args.mask == "3":
        save_dir = f"{save_dir}/mask3"
    if args.mask == "14":
        save_dir = f"{save_dir}/mask14-24"
    if args.mask == "12":
        save_dir = f"{save_dir}/mask12-13-19-23"
    if args.mask == "21":
        save_dir = f"{save_dir}/mask12-13-21-23"
    if args.mask == "12-24":
        save_dir = f"{save_dir}/mask12-24"
        

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    rewards = []
    steps = []
    eval_rewards = []
    timesteps = []
    
    from tqdm import trange
    for i in trange(max_episode):
        with torch.no_grad():
            mb_states, mb_actions, mb_old_a_logps, mb_values, mb_returns, mb_rewards = env_setter.run(env, policy_net, value_net)
            mb_advs = mb_returns - mb_values
            mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-6)

        pg_loss, v_loss, ent = agent.train(mb_states, mb_actions, mb_values, mb_advs, mb_returns, mb_old_a_logps)
        mean_total_reward += mb_rewards.sum()
        mean_length += len(mb_states)
        print("[Episode {:4d}] total reward = {:.6f}, length = {:d}".format(i, mb_rewards.sum(), len(mb_states)))

        rewards.append(mb_rewards.sum())
        steps.append(i)
        
        if i % eval_freq == 0:
            print("\n[{:5d} / {:5d}]".format(i, max_episode))
            print("----------------------------------")
            print("actor loss = {:.6f}".format(pg_loss))
            print("critic loss = {:.6f}".format(v_loss))
            print("entropy = {:.6f}".format(ent))
            print("mean return = {:.6f}".format(mean_total_reward / eval_freq))
            print("mean length = {:.2f}".format(mean_length / eval_freq))

            print("\nSaving the model ... ", end="")
            # torch.save({
            #     "it": i,
            #     "PolicyNet": policy_net.state_dict(),
            #     "ValueNet": value_net.state_dict()
            # }, os.path.join(save_dir, "model.pt"))
            print("Done.\n")

            mean_total_reward = 0
            mean_length = 0

            eval_reward = evaluate(eval_env, policy_net)
            print(f"Evaluation Reward: {eval_reward}")
            eval_rewards.append(eval_reward)
            timesteps.append(i)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards, label="Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Progress Over Time")
    plt.legend()
    plt.grid()
    
    train_reward_path = os.path.join(save_dir, "train_plot.png") if args.mask == "false" else os.path.join(save_dir, "train_plot_masked.png")
    plt.savefig(train_reward_path)
    print(f"Plot saved at: {train_reward_path}")
    
    train_reward_path_np = os.path.join(save_dir, "train_reward.np") if args.mask == "false" else os.path.join(save_dir, "train_reward_masked.np")
    np.savez(train_reward_path_np, timestamps=np.array(steps), rewards=np.array(rewards))
    
    eval_reward_path_np = os.path.join(save_dir, "eval_reward.np") if args.mask == "false" else os.path.join(save_dir, "eval_reward_masked.np")
    np.savez(eval_reward_path_np, timestamps=np.array(timesteps), rewards=np.array(eval_rewards))
    
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, eval_rewards, label="Eval Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Evaluation Reward")
    plt.title("Evaluation Progress Over Time")
    plt.legend()
    plt.grid()
    
    eval_reward_path = os.path.join(save_dir, "eval_plot.png") if args.mask == "false" else os.path.join(save_dir, "eval_plot_masked.png")
    plt.savefig(eval_reward_path)
    print(f"Plot saved at: {eval_reward_path}")



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on BipedalWalker-v3")
    parser.add_argument("--mask", type=str, choices=["false","19","3","12","14","21","12-24"], default="false",
                        help="Apply feature masking: 'true' (removes LIDAR features) or 'false' (full observation).")
    parser.add_argument("--env", type=str, default="BipedalWalker-v3",help="PipedalWalker-v3")
    parser.add_argument("--ckpt", type=str, default="../gail/5000_expert/BipedalWalker-v3",help="checkpoint file name")
    parser.add_argument("--max_episode", type=int, default=5000,help="episode count")
    args = parser.parse_args()
    main(args)