import matplotlib.pyplot as plt
from env import VocRecEnv
from agent import PPOAgent
from utils import ReplayBuffer
import torch
import os
from tqdm import tqdm 

# Ensure the "figure" directory exists for saving plots
if not os.path.exists("figure"):
    os.makedirs("figure")

# Initialize environment, agent, and buffer
env = VocRecEnv()
agent = PPOAgent(env)
buffer = ReplayBuffer()

# Training settings
num_episodes = 50
batch_size = 32

# Lists to store metrics for plotting
actor_losses = []
critic_losses = []
cumulative_rewards = []

# Training loop
for episode in tqdm(range(num_episodes), desc="Training Episodes"):
    state = env.reset()
    episode_reward = 0

    for step in range(100):  # Assuming a maximum of 100 steps per episode
        action, logits = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        # Store experience in buffer
        buffer.add(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state

        # Perform batch update when buffer has enough samples
        if len(buffer.buffer) >= batch_size:
            agent.update(buffer, batch_size)
            # Assume `agent` stores current losses after update for tracking
            actor_losses.append(agent.actor_loss_value)  # Replace with actual actor loss attribute
            critic_losses.append(agent.critic_loss_value)  # Replace with actual critic loss attribute
            buffer.clear()

        if done:
            break

    # Record cumulative reward for this episode
    cumulative_rewards.append(episode_reward)

# Plot Actor and Critic Losses over Episodes
plt.figure(figsize=(10, 5))
plt.plot(range(len(actor_losses)), actor_losses, label="Actor Loss", color="blue")
plt.plot(range(len(critic_losses)), critic_losses, label="Critic Loss", color="red")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Actor and Critic Losses during Training")
plt.legend()
plt.grid(True)
plt.savefig("figure/actor_critic_losses.png")
plt.show()

# Plot Cumulative Rewards over Episodes
plt.figure(figsize=(10, 5))
plt.plot(range(num_episodes), cumulative_rewards, label="Cumulative Reward", color="green")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("figure/cumulative_rewards.png")
plt.show()