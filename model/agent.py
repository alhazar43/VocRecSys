import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
from utils import ReplayBuffer, PPOMemory
from networks import Actor, Critic, ActorCritic
import torch.nn.functional as F
import os



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
        batch_size: int = 64,
        n_epochs: int = 4,
        noise_std: float = 0.1
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.noise_std = noise_std


        self.actor_critic = ActorCritic(n_traits, n_jobs, self.hidden_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        self.memory = PPOMemory(self.batch_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic.to(self.device)
    
    def set_env(self, env):
        """Set environment reference in actor-critic"""
        self.actor_critic.set_env(env)
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Select action and return action probabilities and value"""
        state_tensor = torch.FloatTensor(state).to(self.device)
    
        with torch.no_grad():
            job_scores, action_probs, value = self.actor_critic(state_tensor)
            
            # Add exploration noise to scores
            if self.noise_std > 0:
                noise = torch.randn_like(job_scores) * self.noise_std
                job_scores = job_scores + noise
                # Recompute probabilities with noise
                action_probs = F.softmax(job_scores, dim=-1)
            
        return job_scores.squeeze(0).cpu().numpy(), action_probs, value
    
    def update(self) -> Tuple[float, float, float]:
        # Compute GAE first
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.memory.rewards)).to(self.device)
        values = torch.FloatTensor(np.array(self.memory.vals)).to(self.device)
        
        with torch.no_grad():
            _, _, last_value = self.actor_critic(states[-1])
            advantages = []
            returns = []
            gae = 0
            
            for r, v in zip(reversed(rewards), reversed(values)):
                delta = r + self.gamma * last_value - v
                gae = delta + self.gamma * 0.95 * gae  # 0.95 is GAE lambda
                last_value = v
                advantages.insert(0, gae)
                returns.insert(0, gae + v)
                
            advantages = torch.tensor(advantages).to(self.device)
            returns = torch.tensor(returns).to(self.device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            for batch in self.memory.generate_batches():
                states = torch.FloatTensor(np.array(self.memory.states))[batch].to(self.device)
                old_probs = torch.FloatTensor(np.array(self.memory.probs))[batch].to(self.device)
                old_scores = torch.FloatTensor(np.array(self.memory.actions))[batch].to(self.device)
                rewards = torch.FloatTensor(np.array(self.memory.rewards))[batch].to(self.device)
                next_states = torch.FloatTensor(np.array(self.memory.next_states))[batch].to(self.device)
                values = torch.FloatTensor(np.array(self.memory.vals))[batch].to(self.device)

                # Get current scores, probabilities and values
                new_scores, new_probs, new_values = self.actor_critic(states)
                # _,_, next_values = self.actor_critic(next_states)
                
                # Calculate advantages
                advantages = rewards + self.gamma * new_values.squeeze() - values.squeeze()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Calculate probability ratio
                prob_ratio = (new_probs / (old_probs + 1e-10))
                
                # Calculate surrogate losses
                surr1 = prob_ratio * advantages.unsqueeze(1)
                surr2 = torch.clamp(prob_ratio, 1-self.epsilon, 1+self.epsilon) * advantages.unsqueeze(1)
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_targets = rewards + self.gamma * new_values.squeeze().detach()
                critic_loss = self.c1 * F.mse_loss(new_values.squeeze(), value_targets)
                
                # Calculate entropy bonus
                score_reg = -self.c2 * torch.mean(new_scores)
                total_loss = actor_loss + critic_loss + score_reg
                
                # Update network
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.85)
                self.optimizer.step()
        
        self.memory.clear()
        return actor_loss.item(), critic_loss.item(), score_reg.item()

    # def save(self, path: str):
    #     """Save model, optimizer state, and hyperparameters"""
    #     checkpoint = {
    #         # Model architecture parameters
    #         'n_traits': self.n_traits,
    #         'n_jobs': self.n_jobs,
    #         'hidden_dim': self.hidden_dim,
            
    #         # Model state
    #         'model_state_dict': self.actor_critic.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
            
    #         # Hyperparameters
    #         'gamma': self.gamma,
    #         'epsilon': self.epsilon,
    #         'c1': self.c1,
    #         'c2': self.c2,
    #         'n_epochs': self.n_epochs,
    #         'noise_std': self.noise_std,
    #         'lr': self.lr
    #     }
        
    #     # Create directory if it doesn't exist
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
        
    #     # Save the checkpoint
    #     torch.save(checkpoint, path)
    #     print(f"Model saved to {path}")
    
    # def load(self, path: str):
    #     """Load model, optimizer state, and hyperparameters"""
    #     # Load checkpoint
    #     checkpoint = torch.load(path, map_location=self.device)
        
    #     # Verify model architecture matches
    #     if (checkpoint['n_traits'] != self.n_traits or 
    #         checkpoint['n_jobs'] != self.n_jobs or 
    #         checkpoint['hidden_dim'] != self.hidden_dim):
            
    #         # Recreate model with loaded architecture
    #         self.n_traits = checkpoint['n_traits']
    #         self.n_jobs = checkpoint['n_jobs']
    #         self.hidden_dim = checkpoint['hidden_dim']
    #         self.actor_critic = ActorCritic(self.n_traits, self.n_jobs, self.hidden_dim)
    #         self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=checkpoint['lr'])
        
    #     # Load model state
    #     self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    #     # Load hyperparameters
    #     self.gamma = checkpoint['gamma']
    #     self.epsilon = checkpoint['epsilon']
    #     self.c1 = checkpoint['c1']
    #     self.c2 = checkpoint['c2']
    #     self.n_epochs = checkpoint['n_epochs']
    #     self.noise_std = checkpoint['noise_std']
    #     self.lr = checkpoint['lr']
        
    #     # Move model to correct device
    #     self.actor_critic.to(self.device)
    #     print(f"Model loaded from {path}")
    
    # @classmethod
    # def load_from_checkpoint(cls, path: str):
    #     """Create a new agent from a checkpoint file"""
    #     # Load checkpoint
    #     checkpoint = torch.load(path)
        
    #     # Create new agent with loaded parameters
    #     agent = cls(
    #         n_traits=checkpoint['n_traits'],
    #         n_jobs=checkpoint['n_jobs'],
    #         hidden_dim=checkpoint['hidden_dim'],
    #         lr=checkpoint['lr'],
    #         gamma=checkpoint['gamma'],
    #         epsilon=checkpoint['epsilon'],
    #         c1=checkpoint['c1'],
    #         c2=checkpoint['c2'],
    #         n_epochs=checkpoint['n_epochs'],
    #         noise_std=checkpoint['noise_std']
    #     )
        
    #     # Load state dictionaries
    #     agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
    #     agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    #     return agent