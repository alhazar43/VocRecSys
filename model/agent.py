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

        self.actor_loss_value = 0.0  # Initialize actor loss value
        self.critic_loss_value = 0.0  # Initialize critic loss value
        
        sample_observation = env.reset()
        input_dim = sample_observation.shape[0] * sample_observation.shape[1]

        self.actor = ActorNet(input_dim=input_dim, hidden_dim=self.hidden_dim, n_jobs=env.n_jobs)
        self.critic = CriticNet(input_dim=input_dim, hidden_dim=self.hidden_dim)
        
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
        state = torch.FloatTensor(state).unsqueeze(0)

        logits = self.actor(state)
        # Use logits to sample a ranking of courses (full permutation)
        # action_dist = Categorical(logits=logits)
        # ranking = torch.argsort(action_dist.sample(), descending=False).cpu().numpy()
        probs = torch.softmax(logits, dim=-1).squeeze() 
        ranking = torch.argsort(probs, descending=True).cpu().numpy()

        return ranking, probs.cpu().detach().numpy()

    # def update(self, buffer, batch_size=32):
    #     # Sample a batch from the buffer
    #     batch = buffer.sample(batch_size)

    #     # Unpack each component of the sampled batch correctly
    #     # We assume each batch entry is a tuple of (state, action, reward, next_state, done)
    #     states = torch.FloatTensor([entry[0] for entry in batch])  # Extract states as-is
    #     actions = torch.LongTensor([entry[1] for entry in batch])  # Actions
    #     rewards = torch.FloatTensor([entry[2] for entry in batch])  # Rewards
    #     next_states = torch.FloatTensor([entry[3] for entry in batch])  # Next states as-is
    #     dones = torch.FloatTensor([entry[4] for entry in batch])  # Done flags

    #     # Critic update
    #     current_values = self.critic(states).squeeze()
    #     next_values = self.critic(next_states).squeeze()
    #     expected_values = rewards + self.gamma * next_values * (1 - dones)
        
    #     critic_loss = F.mse_loss(current_values, expected_values.detach())
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()

    #     # Actor update using PPO
    #     logits = self.actor(states)
    #     action_probs = torch.softmax(logits, dim=-1)
    #     action_dist = Categorical(action_probs)

    #     # Adjust dimensions of actions to match action_probs for torch.gather
    #     # actions = actions.view(-1, 1)  # Ensure actions is 2D
    #     old_action_probs = action_probs.gather(1, actions)

    #     # Calculate advantage without in-place operations
    #     current_values = self.critic(states).squeeze()
    #     next_values = self.critic(next_states).squeeze()
    #     expected_values = rewards + self.gamma * next_values * (1 - dones)
    #     advantage = (expected_values - current_values).view(-1, 1)

    #     # Calculate ratio and surrogate losses without in-place modifications
    #     ratio = torch.exp(torch.log(action_probs.gather(1, actions)) - torch.log(old_action_probs))
    #     surr1 = ratio * advantage
    #     surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
    #     actor_loss = -torch.min(surr1, surr2).mean()

    #     # Perform actor and critic optimization steps
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()  # Ensure this is not modified in place
    #     self.actor_optimizer.step()

    #     # Critic loss and step
    #     critic_loss = F.mse_loss(current_values, expected_values.detach())
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()  # Ensure no in-place modification
    #     self.critic_optimizer.step()

    def update(self, buffer, batch_size=32):
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
