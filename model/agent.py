from network import ActorNet, CriticNet
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PPOAgent:
    def __init__(self, env, hidden_dim=128, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, epsilon=0.2):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.actor_loss_value = 0.0  # Initialize actor loss value
        self.critic_loss_value = 0.0  # Initialize critic loss value
        
        sample_observation = env.reset()
        input_dim = sample_observation.shape[0] * sample_observation.shape[1]

        self.actor = ActorNet(input_dim=input_dim, hidden_dim=self.hidden_dim, n_jobs=env.n_jobs)
        self.critic = CriticNet(input_dim=input_dim, hidden_dim=self.hidden_dim)

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    # def select_action(self, state):
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     logits = self.actor(state)
        
    #     # Use logits to sample a ranking of courses (full permutation)
    #     action_dist = Categorical(logits=logits)
    #     ranking = torch.argsort(action_dist.sample(), descending=False).cpu().numpy()
    #     return ranking, logits
    def select_action(self, state):
        self.actor.eval()
        # Convert single state to tensor
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        
        logits = self.actor(state)
        probs = torch.softmax(logits, dim=-1).squeeze()
        ranking = torch.argsort(probs, descending=True).cpu().detach().numpy()
        
        self.actor.train()
        return ranking, probs.cpu().detach().numpy()

    def update(self, buffer, batch_size=32):
        if len(buffer.buffer) < batch_size:
            return
            
        batch = buffer.sample(batch_size)
        
        # Convert batch data to numpy arrays
        states = np.array([item[0] for item in batch])
        actions = np.array([item[1] for item in batch])  # This should be indices
        rewards = np.array([item[2] for item in batch])
        next_states = np.array([item[3] for item in batch])
        dones = np.array([item[4] for item in batch])
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        # Ensure actions are properly shaped for gathering
        actions = torch.LongTensor(actions).reshape(-1, 1).to(self.device)  # Shape: [batch_size, 1]
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute values
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()
        
        current_values = self.critic(states).squeeze()
        
        # Compute advantages
        expected_values = rewards + self.gamma * next_values * (1 - dones)
        advantage = (expected_values - current_values.detach()).unsqueeze(-1)
        
        # Actor update
        logits = self.actor(states)  # Shape: [batch_size, n_jobs]
        action_probs = torch.softmax(logits, dim=-1)  # Shape: [batch_size, n_jobs]
        
        # Gather the probabilities for the actions that were taken
        old_action_probs = action_probs.gather(1, actions)  # Shape: [batch_size, 1]
        
        # Compute PPO ratio
        new_action_probs = action_probs.gather(1, actions)
        ratio = torch.exp(torch.log(new_action_probs) - torch.log(old_action_probs))
        
        # Compute surrogate losses
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        # Optimize critic
        critic_loss = F.mse_loss(current_values, expected_values.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        # Store loss values
        self.actor_loss_value = actor_loss.item()
        self.critic_loss_value = critic_loss.item()

    def update_dep(self, buffer, batch_size=32):
        # Sample a batch from the buffer and process components
        batch = buffer.sample(batch_size)

        states = torch.FloatTensor([entry[0] for entry in batch])
        actions = torch.LongTensor([entry[1] for entry in batch])
        rewards = torch.FloatTensor([entry[2] for entry in batch])
        next_states = torch.FloatTensor([entry[3] for entry in batch])
        dones = torch.FloatTensor([entry[4] for entry in batch])

        # Forward pass through actor network to get action probabilities
        logits = self.actor(states)
        action_probs = torch.softmax(logits, dim=-1)
        
        # Gather probabilities for the specific actions taken, along dim=1
        old_action_probs = action_probs.gather(1, actions)

        # Critic values for current and next states
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()  # Detach to avoid graph overlap

        current_values = self.critic(states).squeeze()  # Keep this in graph only for critic_loss

        # Calculate advantage completely detached from the graph
        expected_values = (rewards + self.gamma * next_values * (1 - dones))
        advantage = (expected_values - current_values.detach()).view(-1, 1)

        # Actor loss calculation
        ratio = torch.exp(torch.log(action_probs.gather(1, actions)) - torch.log(old_action_probs))
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        # Actor optimization step
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_loss_value = actor_loss.item()

        # Critic loss and optimization step, using detached expected values
        critic_loss = F.mse_loss(current_values, expected_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_loss_value = critic_loss.item()
