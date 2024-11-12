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