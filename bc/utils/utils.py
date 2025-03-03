# utils.py
import torch
from torch.utils.data import Dataset
import numpy as np

def to_input(states, actions, n=2, compare=1):
    '''
    Data preparation and filtering 
    Inputs:
    states: expert states as tensor
    actions: actions states as tensor
    n: window size (how many states needed to predict the next action)
    compare: for filtering data 
    return:
    output_states: filtered states as tensor 
    output_actions: filtered actions as tensor 
    '''
    count = 0
    index = []
    ep, t, state_size = states.shape
    _, _, action_size = actions.shape

    # Check if the window size n is valid
    if n > t:
        raise ValueError(f"Window size n={n} is greater than the number of timesteps t={t}.")

    output_states = torch.zeros((ep * (t - n + 1), state_size * n), dtype=torch.float)
    output_actions = torch.zeros((ep * (t - n + 1), action_size), dtype=torch.float)

    for i in range(ep):
        for j in range(t - n + 1):
            # Ensure the comparison results in a tensor
            if torch.all(states[i, j] == -compare * torch.ones_like(states[i, j])):
                index.append([i, j])
            else:
                output_states[count] = states[i, j:j + n].view(-1)
                output_actions[count] = actions[i, j]
                count += 1

    output_states = output_states[:count]
    output_actions = output_actions[:count]

    return output_states, output_actions

class ExpertDataset(Dataset):
    """Dataset for expert trajectories."""

    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]