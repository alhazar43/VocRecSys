import numpy as np
import torch
from typing import Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from env import VocRecEnv
from agent_new import PPOAgent

class Trainer:
    def __init__(
        self,
        env_kwargs: Dict = None,
        agent_kwargs: Dict = None,
        save_dir: str = 'results',
        figure_dir: str = 'figure'
    ):
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)
        
        self.save_dir = save_dir
        self.figure_dir = figure_dir
        
        # Initialize environment
        env_kwargs = env_kwargs or {}
        self.env = VocRecEnv(**env_kwargs)
        
        # Initialize agent
        agent_kwargs = agent_kwargs or {}
        self.agent = PPOAgent(
            n_traits=self.env.n_traits,
            n_jobs=self.env.n_jobs,
            **agent_kwargs
        )
        
        # Initialize metrics storage
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.avg_job_scores = []
        
    def train(
        self,
        n_episodes: int,
        steps_per_episode: int,
        update_frequency: int = 5,
        log_frequency: int = 10,
        save_frequency: int = 100
    ):
        """Train the agent"""
        print(f"\nStarting training for {n_episodes} episodes...")
        
        # Main training loop
        episode_pbar = tqdm(range(n_episodes), desc="Episodes")
        for episode in episode_pbar:
            state = self.env.reset()
            episode_reward = 0
            episode_actor_losses = []
            episode_critic_losses = []
            episode_scores = []
            
            # Episode loop
            for step in range(steps_per_episode):
                # Select action
                scores, _ = self.agent.select_action(state)
                episode_scores.append(scores.mean())
                
                # Take step in environment
                next_state, reward, done, _ = self.env.step(scores)
                
                # Store transition
                self.agent.replay_buffer.push(
                    state, scores, reward, next_state
                )
                
                state = next_state
                episode_reward += reward
                
                # Update policy
                if (episode * steps_per_episode + step + 1) % update_frequency == 0:
                    actor_loss, critic_loss, _ = self.agent.update()
                    episode_actor_losses.append(actor_loss)
                    episode_critic_losses.append(critic_loss)
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            if episode_actor_losses:
                self.actor_losses.extend(episode_actor_losses)
                self.critic_losses.extend(episode_critic_losses)
            self.avg_job_scores.append(np.mean(episode_scores))
            
            # Update progress bar
            episode_pbar.set_postfix({
                'reward': f'{episode_reward:.3f}',
                'avg_100': f'{np.mean(self.episode_rewards[-100:]):.3f}' if len(self.episode_rewards) >= 100 else 'N/A',
                'avg_score': f'{np.mean(episode_scores):.3f}'
            })
            
            # Log and save
            if (episode + 1) % log_frequency == 0:
                self.plot_metrics()
            
            if (episode + 1) % save_frequency == 0:
                self.save_checkpoint(episode + 1)
        
        # Final plots and save
        self.plot_metrics()
        self.save_checkpoint('final')
        print("\nTraining completed!")
        
    def plot_metrics(self):
        """Plot and save training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics')
        
        # Plot rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) >= 99:
            # smoothed = np.convolve(self.episode_rewards, 
            #                      np.ones(99)/100, 
            #                      mode='valid')
            smoothed = self.moving_average(self.episode_rewards, window_size=100, shrink=True)
            ax.plot(smoothed, label='Smoothed (100 ep)')
        ax.set_title('Episode Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True)
        
        # Plot actor loss
        ax = axes[0, 1]
        if self.actor_losses:
            ax.plot(self.actor_losses)
        ax.set_title('Actor Loss')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
        ax.grid(True)
        
        # Plot critic loss
        ax = axes[1, 0]
        if self.critic_losses:
            ax.plot(self.critic_losses)
        ax.set_title('Critic Loss')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
        ax.grid(True)
        
        # Plot average job scores
        ax = axes[1, 1]
        ax.plot(self.avg_job_scores)
        ax.set_title('Average Job Scores')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, 'training_metrics.png'))
        plt.close()
    
    @staticmethod
    def moving_average(data, window_size, shrink=False):
        """Compute the moving average with a given window size
           If shrink is True, the window size will reduce at the end to avoid blanks."""
        if not shrink:
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        else:
            return [np.mean(data[max(0, i - window_size + 1):i + 1]) for i in range(len(data))]

    def save_checkpoint(self, episode):
        """Save a training checkpoint"""
        checkpoint = {
            'episode': episode,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'actor_optimizer_state_dict': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
            'metrics': {
                'episode_rewards': self.episode_rewards,
                'actor_losses': self.actor_losses,
                'critic_losses': self.critic_losses
            }
        }
        path = os.path.join(self.save_dir, f'checkpoint_ep{episode}.pt')
        torch.save(checkpoint, path)
        print(f"\nSaved checkpoint to {path}")

if __name__ == "__main__":
    # Training configuration
    env_kwargs = {
        'n_traits': 6,
        'n_jobs': 100,
        'top_k': 5
    }
    
    agent_kwargs = {
        'hidden_dim': 64,
        'lr': 3e-3,
        'gamma': 0.95,
        'epsilon': 0.1,
        'c1': 0.75,  # Value loss coefficient
        'c2': 0.01,
        'buffer_size': 10000,
        'batch_size': 128
    }
    
    # Create trainer
    trainer = Trainer(
        env_kwargs=env_kwargs,
        agent_kwargs=agent_kwargs,
        save_dir='results',
        figure_dir='figure'
    )
    
    # Start training
    trainer.train(
        n_episodes=300,
        steps_per_episode=300,
        update_frequency=5
        ,
        log_frequency=10,
        save_frequency=100
    )