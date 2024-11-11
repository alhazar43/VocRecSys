
import numpy as np
import math
import random
import torch
from torch.utils.data import DataLoader, Dataset



class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size
        self._ptr = 0
        
    def add(self, state, action, reward, next_state, done):
        # Ensure action is a scalar index
        if isinstance(action, np.ndarray):
            action = action.item() if action.size == 1 else action[0]
            
        state = np.array(state)
        next_state = np.array(next_state)
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self._ptr] = (state, action, reward, next_state, done)
        self._ptr = (self._ptr + 1) % self.buffer_size
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]
    def clear(self):
        self.buffer = []
        self._ptr = 0
        
        
    def __len__(self):
        return len(self.buffer)