from network import ActorNet, CriticNet
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Categorical
import numpy as np

class PPOAgent:
    def __init__(self, env, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, epsilon=0.2):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        input_dim = env.n_jobs + 1 
        self.actor = ActorNet(input_dim, env.n_jobs)
        self.critic = CriticNet(input_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.actor(state)
        
        # Use logits to sample a ranking of courses (full permutation)
        action_dist = Categorical(logits=logits)
        ranking = torch.argsort(action_dist.sample(), descending=False).cpu().numpy()
        return ranking, logits

    def update(self, buffer, batch_size=32):
        # Sample a batch from the buffer
        states, actions, rewards, next_states, dones = zip(*buffer.sample(batch_size))

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Update Critic
        current_values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        expected_values = rewards + self.gamma * next_values * (1 - dones)

        critic_loss = F.mse_loss(current_values, expected_values.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor using PPO
        logits = self.actor(states)
        action_probs = torch.softmax(logits, dim=-1)
        action_dist = Categorical(action_probs)
        
        old_action_probs = torch.gather(action_probs, 1, actions.unsqueeze(1)).squeeze()
        
        advantage = (rewards + self.gamma * next_values * (1 - dones)) - current_values
        ratio = torch.exp(torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze()) - torch.log(old_action_probs))

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()