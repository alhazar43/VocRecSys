import numpy as np
from typing import Tuple
from collections import deque
import random
import torch

class ReplayBuffer:
    """Replay buffer for storing transitions"""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, 
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray
    ) -> None:
        """Store transition in buffer"""
        self.buffer.append((state, action, reward, next_state))
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of transitions"""
        transitions = random.sample(self.buffer, batch_size)
        batch = list(zip(*transitions))
        
        states = torch.FloatTensor(np.stack(batch[0]))
        actions = torch.FloatTensor(np.stack(batch[1]))
        rewards = torch.FloatTensor(batch[2])
        next_states = torch.FloatTensor(np.stack(batch[3]))
        
        return states, actions, rewards, next_states
    
    def __len__(self) -> int:
        return len(self.buffer)

    def get_recent_rewards(self, n: int = 100) -> np.ndarray:
        """Get the n most recent rewards"""
        n = min(n, len(self.buffer))
        recent_transitions = list(self.buffer)[-n:]
        return np.array([t[2] for t in recent_transitions])
    

class PPOMemory:
    """Memory buffer for storing trajectories"""
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []    # Store old action probabilities
        self.vals = []     # Store values
        self.rewards = []
        self.next_states = []
        self.batch_size = batch_size

    def store(self, state, action, prob, val, reward, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.next_states = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches