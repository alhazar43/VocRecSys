import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
from utils import ReplayBuffer
from networks import Actor, Critic
import torch.nn.functional as F


class PPOAgent:
    def __init__(
        self,
        n_traits: int,
        n_jobs: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        c1: float = 0.5,
        c2: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 64,
        noise_std: float = 0.1
    ):
        self.n_traits = n_traits
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.noise_std = noise_std
        
        self.actor = Actor(n_traits, n_jobs, hidden_dim)
        self.critic = Critic(n_traits, hidden_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def to(self, device: torch.device):
        """Move networks to device"""
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Select action with exploration noise"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            scores = self.actor(state_tensor)
            # Add exploration noise
            scores = F.softplus(scores)  # Ensure positive scores
            
            # Add exploration noise
            noise = torch.randn_like(scores) * self.noise_std
            scores = scores + noise
            scores = torch.clamp(scores, min=1e-6)  # Prevent zero scores

            scores = scores / scores.sum()
        return scores.cpu().numpy(), torch.zeros(1)  # dummy log_prob for compatibility
    
    def update(self) -> Tuple[float, float, float]:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0
        
        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        
        # Compute advantages
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Get current policy distribution
        current_scores = self.actor(states)
        current_scores = F.softmax(current_scores, dim=1)
        
        # Compute value estimates
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            advantages = rewards + self.gamma * next_values - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = rewards + self.gamma * next_values
        
        # Compute probability ratio using KL divergence
        old_scores = F.softmax(actions, dim=1)
        ratio = torch.exp(torch.sum(current_scores.log() * old_scores, dim=1))
        
        # PPO losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Add entropy bonus for exploration
        entropy = -(current_scores * torch.log(current_scores + 1e-8)).sum(dim=1).mean()
        actor_loss = actor_loss - self.c2 * entropy
        
        # Value loss with clipping
        value_pred = self.critic(states).squeeze()
        value_loss = F.smooth_l1_loss(value_pred, returns)  # Use Huber loss
        critic_loss = self.c1 * value_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), entropy.item()

    def save(self, path: str):
        """Save model state"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])