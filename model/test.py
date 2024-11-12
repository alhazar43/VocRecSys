import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
from tqdm import tqdm
from env import VocRecEnv
from datetime import datetime
from agent import PPOAgent

class Trainer:
    def __init__(
        self,
        agent,
        env,
        save_dir: str = 'checkpoints',
        fig_dir: str = 'figures'
    ):
        self.agent = agent
        self.env = env
        
        # Create directories
        self.save_dir = save_dir
        self.fig_dir = fig_dir
        for dir_path in [save_dir, fig_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize metrics storage
        self.episode_rewards: List[float] = []
        self.avg_rewards: List[float] = []
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        
        # Training state
        self.current_episode = 0
        self.best_avg_reward = float('-inf')
        
        # Create unique run identifier
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def save_checkpoint(self, path: str):
        """Save complete training state"""
        checkpoint = {
            # Model and optimizer states
            'model_state': self.agent.actor_critic.state_dict(),
            'optimizer_state': self.agent.optimizer.state_dict(),
            
            # Training metrics
            'episode_rewards': self.episode_rewards,
            'avg_rewards': self.avg_rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            
            # Training state
            'current_episode': self.current_episode,
            'best_avg_reward': self.best_avg_reward,
            'run_id': self.run_id,
            
            # Environment parameters
            'env_params': {
                'n_traits': self.env.n_traits,
                'n_jobs': self.env.n_jobs,
                'top_k': self.env.K,
                'ability_range': self.env.ability_range,
                'n_clusters': self.env.n_clusters
            },
            
            # Agent parameters
            'agent_params': {
                'hidden_dim': self.agent.hidden_dim,
                'lr': self.agent.lr,
                'gamma': self.agent.gamma,
                'epsilon': self.agent.epsilon,
                'c1': self.agent.c1,
                'c2': self.agent.c2,
                'batch_size': self.agent.batch_size,
                'n_epochs': self.agent.n_epochs,
                'noise_std': self.agent.noise_std
            }
        }
        
        torch.save(checkpoint, path)
        print(f"\nCheckpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, agent_class, env_class):
        """
        Load complete training state and recreate trainer
        Returns: trainer, remaining_episodes
        """
        # Load checkpoint
        checkpoint = torch.load(path)
        
        # Recreate environment
        env = env_class(**checkpoint['env_params'])
        
        # Recreate agent
        agent = agent_class(
            n_traits=env.n_traits,
            n_jobs=env.n_jobs,
            **checkpoint['agent_params']
        )
        
        # Load model and optimizer states
        agent.actor_critic.load_state_dict(checkpoint['model_state'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Create trainer
        trainer = cls(agent=agent, env=env)
        
        # Load training state
        trainer.episode_rewards = checkpoint['episode_rewards']
        trainer.avg_rewards = checkpoint['avg_rewards']
        trainer.actor_losses = checkpoint['actor_losses']
        trainer.critic_losses = checkpoint['critic_losses']
        trainer.current_episode = checkpoint['current_episode']
        trainer.best_avg_reward = checkpoint['best_avg_reward']
        trainer.run_id = checkpoint['run_id']
        
        return trainer
        
    def train(
        self,
        n_episodes: int,
        steps_per_episode: int,
        eval_freq: int = 10,
        save_freq: int = 100,
        update_freq: int = 10,
        continue_training: bool = False
    ):
        """Train the agent"""
        start_episode = self.current_episode if continue_training else 0
        eval_window = 100
        
        # Training loop
        progress_bar = tqdm(range(start_episode, n_episodes), desc="Training")
        for episode in progress_bar:
            state = self.env.reset()
            episode_reward = 0
            episode_actor_losses = []
            episode_critic_losses = []
            
            # Episode loop
            for step in range(steps_per_episode):
                # Select action
                action_array, action_probs, value = self.agent.select_action(state)
                
                # Take environment step
                next_state, reward, done, _ = self.env.step(action_array)
                episode_reward += reward
                
                # Store transition
                self.agent.memory.store(
                    state, action_array, action_probs.detach().cpu().numpy(),
                    value.detach().cpu().numpy(), reward, next_state
                )
                
                state = next_state
                
                # Update agent
                if step > 0 and step % update_freq == 0:
                    actor_loss, critic_loss, _ = self.agent.update()
                    episode_actor_losses.append(actor_loss)
                    episode_critic_losses.append(critic_loss)
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            if episode_actor_losses:
                self.actor_losses.extend(episode_actor_losses)
                self.critic_losses.extend(episode_critic_losses)
            
            # Calculate moving average
            recent_rewards = self.episode_rewards[-eval_window:] if len(self.episode_rewards) >= eval_window else self.episode_rewards
            avg_reward = np.mean(recent_rewards)
            self.avg_rewards.append(avg_reward)
            
            # Update progress bar
            progress_bar.set_postfix({
                'reward': f'{episode_reward:.2f}',
                'avg_reward': f'{avg_reward:.2f}'
            })
            
            # Save best model
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.save_checkpoint(f"{self.save_dir}/best_model_{self.run_id}.pt")
            
            # Periodic saving and plotting
            if (episode + 1) % save_freq == 0:
                self.save_checkpoint(f"{self.save_dir}/checkpoint_ep{episode+1}_{self.run_id}.pt")
            
            if (episode + 1) % eval_freq == 0:
                self.plot_metrics()
            
            self.current_episode = episode + 1
        
        # Final plots and saves
        self.plot_metrics()
        self.save_checkpoint(f"{self.save_dir}/final_model_{self.run_id}.pt")
        print(f"\nTraining completed! Best average reward: {self.best_avg_reward:.2f}")

    def plot_metrics(self):
        """Plot and save training metrics"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Training Metrics (Episodes 1-{self.current_episode})')
        
        # Plot rewards
        ax = axes[0]
        ax.plot(self.episode_rewards, alpha=0.6, label='Episode Rewards', color='blue')
        ax.plot(self.avg_rewards, label='Average Reward', color='red', linewidth=2)
        ax.set_title('Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True)
        
        # Plot losses
        ax = axes[1]
        if self.actor_losses and self.critic_losses:
            ax.plot(self.actor_losses, label='Actor Loss', alpha=0.6)
            ax.plot(self.critic_losses, label='Critic Loss', alpha=0.6)
            ax.set_title('Losses')
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/training_metrics_{self.run_id}.png")
        plt.close()

if __name__ == "__main__":
    
    # Create new training run
    env_kwargs = {
        'n_traits': 6,
        'n_jobs': 100,
        'top_k': 20,
        'ability_range': (-3, 3),
        'n_clusters': 5
    }
    
    agent_kwargs = {
        'hidden_dim': 128,
        'lr': 3e-4,
        'gamma': 0.95,
        'epsilon': 0.1,
        'c1': 0.5,
        'c2': 0.05,
        'batch_size': 64,
        'n_epochs': 4,
        'noise_std': 0.1
    }
    
    # Either start new training:

    env = VocRecEnv(**env_kwargs)
    agent = PPOAgent(n_traits=env.n_traits, n_jobs=env.n_jobs, **agent_kwargs)
    agent.set_env(env)  # Add this line
    trainer = Trainer(agent=agent, env=env)
    
    # Or load from checkpoint:
    # trainer = Trainer.load_checkpoint(
    #     "checkpoints/checkpoint_ep500_20240312_123456.pt",
    #     PPOAgent,
    #     VocRecEnv
    # )
    
    train_kwargs = {
        'n_episodes': 500,
        'steps_per_episode': 200,
        'eval_freq': 10,
        'save_freq': 100,
        'update_freq': 10,
        'continue_training': False  # Set to True if loading from checkpoint
    }
    
    trainer.train(**train_kwargs)