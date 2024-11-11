import numpy as np
from IRT import AdaptiveMIRT
from typing import Tuple, Dict, Any
from numba import jit
import time
from tqdm import tqdm

class VocRecEnv2:
    """
    Optimized environment for job recommendation with MIRT-based ability estimation.
    Maintains consistent rewards with original implementation while improving performance.
    """
    def __init__(
        self, 
        n_traits: int = 6,
        n_jobs: int = 100,
        top_k: int = 5,
        ability_range: Tuple[float, float] = (-3, 3)
    ):
        self.n_traits = n_traits
        self.n_jobs = n_jobs
        self.K = top_k
        self.ability_range = ability_range
        self.mirt = AdaptiveMIRT(n_traits=n_traits)
        
        # Initialize as contiguous arrays
        self.job_reqs = np.ascontiguousarray(
            np.random.uniform(
                ability_range[0], 
                ability_range[1], 
                size=(n_jobs, n_traits)
            ), 
            dtype=np.float32
        )
        
        # Pre-allocate arrays
        self.current_ability = None
        self._match_qualities = np.zeros(n_jobs, dtype=np.float32)
        self._top_k_indices = np.zeros(top_k, dtype=np.int32)
        self._abs_diffs = np.zeros((n_jobs, n_traits), dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reset environment state."""
        self.mirt = AdaptiveMIRT(n_traits=self.n_traits)
        self.current_ability = self.mirt._get_theta()
        
        # Generate new job requirements
        rng = np.random.default_rng()
        self.job_reqs = np.ascontiguousarray(
            rng.uniform(
                self.ability_range[0],
                self.ability_range[1],
                size=(self.n_jobs, self.n_traits)
            ).astype(np.float32)
        )
        return self.current_ability

    def _compute_reward(self, scores: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Optimized reward computation that maintains consistency with original version.
        Returns both reward and top-k indices.
        """
        # Get top-k indices using partition
        top_k_indices = np.argpartition(scores, -self.K)[-self.K:]
        np.copyto(self._top_k_indices, top_k_indices)
        
        # Compute absolute differences
        ability_expanded = np.expand_dims(self.current_ability, 0)
        np.subtract(self.job_reqs, ability_expanded, out=self._abs_diffs)
        np.abs(self._abs_diffs, out=self._abs_diffs)
        
        # Compute mean differences along traits axis
        np.mean(self._abs_diffs, axis=1, out=self._match_qualities)
        
        # Convert to match qualities (1 - mean_diff)
        np.subtract(1.0, self._match_qualities, out=self._match_qualities)
        
        # Compute reward for top-k jobs
        reward = float(np.mean(self._match_qualities[top_k_indices]))
        
        return reward, self._top_k_indices

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take environment step."""
        if len(action) != self.n_jobs:
            raise ValueError(f"Action must be length {self.n_jobs}, got {len(action)}")
        
        # Ensure action is float32
        if action.dtype != np.float32:
            action = action.astype(np.float32, copy=False)
        
        # Compute reward and get top-k indices
        reward, top_k_indices = self._compute_reward(action)
        
        # Update MIRT state
        _ = self.mirt.next_item()
        self.mirt.sim_resp()
        self.current_ability = self.mirt._get_theta()
        
        return self.current_ability, reward, False, {'top_k_jobs': top_k_indices}
    



def run_benchmark(n_episodes: int = 1000, n_steps: int = 100):
    """Run extensive benchmark"""
    env = VocRecEnv2()
    
    # Storage for timing data
    reset_times = []
    step_times = []
    rewards = []
    episode_total_times = []
    
    # Main episode loop with tqdm
    for _ in tqdm(range(n_episodes), desc="Episodes", ncols=100):
        episode_start = time.perf_counter()
        
        # Time reset
        t0 = time.perf_counter()
        state = env.reset()
        reset_times.append(time.perf_counter() - t0)
        
        episode_rewards = []
        
        # Run steps
        for _ in range(n_steps):
            action = np.random.rand(env.n_jobs)
            
            t0 = time.perf_counter()
            next_state, reward, done, info = env.step(action)
            step_times.append(time.perf_counter() - t0)
            
            episode_rewards.append(reward)
        
        rewards.extend(episode_rewards)
        episode_total_times.append(time.perf_counter() - episode_start)
    
    # Convert timing data to milliseconds
    reset_times = np.array(reset_times) * 1000
    step_times = np.array(step_times) * 1000
    episode_total_times = np.array(episode_total_times) * 1000
    rewards = np.array(rewards)
    
    # Compute statistics
    stats = {
        'total_time_seconds': sum(episode_total_times) / 1000,
        'reset_time': {
            'mean': np.mean(reset_times),
            'std': np.std(reset_times),
            'min': np.min(reset_times),
            'max': np.max(reset_times),
            'p50': np.percentile(reset_times, 50),
            'p95': np.percentile(reset_times, 95),
        },
        'step_time': {
            'mean': np.mean(step_times),
            'std': np.std(step_times),
            'min': np.min(step_times),
            'max': np.max(step_times),
            'p50': np.percentile(step_times, 50),
            'p95': np.percentile(step_times, 95),
        },
        'episode_time': {
            'mean': np.mean(episode_total_times),
            'std': np.std(episode_total_times),
            'min': np.min(episode_total_times),
            'max': np.max(episode_total_times),
            'p50': np.percentile(episode_total_times, 50),
            'p95': np.percentile(episode_total_times, 95),
        },
        'rewards': {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'p50': np.percentile(rewards, 50),
            'p95': np.percentile(rewards, 95),
        }
    }
    
    return stats

def print_stats(stats: Dict):
    """Print benchmark statistics in a readable format"""
    print("\nBenchmark Results")
    print("=" * 50)
    
    print(f"\nTotal Runtime: {stats['total_time_seconds']:.2f} seconds")
    
    metrics = ['reset_time', 'step_time', 'episode_time']
    names = ['Reset Time (ms)', 'Step Time (ms)', 'Episode Time (ms)']
    
    for metric, name in zip(metrics, names):
        print(f"\n{name}")
        print("-" * 40)
        print(f"Mean ± Std: {stats[metric]['mean']:.3f} ± {stats[metric]['std']:.3f}")
        print(f"Min: {stats[metric]['min']:.3f}")
        print(f"Max: {stats[metric]['max']:.3f}")
        print(f"Median: {stats[metric]['p50']:.3f}")
        print(f"95th percentile: {stats[metric]['p95']:.3f}")
    
    print("\nRewards")
    print("-" * 40)
    print(f"Mean ± Std: {stats['rewards']['mean']:.3f} ± {stats['rewards']['std']:.3f}")
    print(f"Min: {stats['rewards']['min']:.3f}")
    print(f"Max: {stats['rewards']['max']:.3f}")
    print(f"Median: {stats['rewards']['p50']:.3f}")
    print(f"95th percentile: {stats['rewards']['p95']:.3f}")

if __name__ == "__main__":
    # Optional: enable numpy optimizations
    np.seterr(all='ignore')  # Ignore numpy warnings
    
    # Run benchmark
    stats = run_benchmark(n_episodes=1000, n_steps=100)
    
    # Print results
    print_stats(stats)
    
    # Save results
    try:
        import json
        from datetime import datetime
        
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'env_version': 'VocRecEnv2',
            'stats': stats
        }
        
        with open('benchmark_results_vocrecenv2.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to 'benchmark_results_vocrecenv2.json'")
    except:
        print("\nFailed to save results to file")