
import numpy as np

from agent import PPOAgent
from env import VocRecEnv
from utils import ReplayBuffer
from tqdm.auto import tqdm
import os
import random
import matplotlib.pyplot as plt


import numpy as np
env = VocRecEnv()
agent = PPOAgent(env)
buffer = ReplayBuffer()

num_episodes = 1000
batch_size = 32

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(100):  # Assuming a maximum of 100 steps per episode
        action, logits = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        buffer.add(state, action, reward, next_state, done)
        episode_reward += reward

        state = next_state

        if len(buffer.buffer) >= batch_size:
            agent.update(buffer, batch_size)
            buffer.clear()

        if done:
            break

    print(f"Episode {episode + 1}, Total Reward: {episode_reward}")