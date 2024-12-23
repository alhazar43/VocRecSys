import numpy as np
import time
from typing import Tuple, Dict, Any
from IRT import AdaptiveMIRT
from tqdm import tqdm
from sklearn.datasets import make_blobs

def generate_job_reqs(n_jobs, n_traits, n_clusters, ability_range, cluster_std=1.2):
    # Generate the cluster centers within the desired range
    cluster_positions = np.linspace(ability_range[0], ability_range[1], n_clusters)
    centers = np.array([
        [pos] * n_traits for pos in cluster_positions
    ])
    
    # Generate the data points
    X, _ = make_blobs(
        n_samples=n_jobs,
        n_features=n_traits,
        centers=centers,  # Set the range for the cluster centers
        cluster_std=cluster_std,  # Standard deviation of clusters
        random_state=42
    )

    X = np.clip(X, ability_range[0], ability_range[1])

    
    return X


class VocRecEnv:
    """
    Optimized environment for job recommendation with MIRT-based ability estimation.
    Maintains consistent rewards with original implementation while improving performance.
    """
    def __init__(
        self, 
        n_traits: int = 6,
        n_jobs: int = 100,
        top_k: int = 20,
        ability_range: Tuple[float, float] = (-3, 3),
        n_clusters: int = 5,
        diversity_weight: float = 0.3
    ):
        self.n_traits = n_traits
        self.n_jobs = n_jobs
        self.K = top_k
        self.diversity_weight = diversity_weight
        self.ability_range = ability_range
        self.n_clusters = n_clusters
        self.mirt = AdaptiveMIRT(n_traits=n_traits)
        
        # Initialize as contiguous arrays
        self.job_reqs = np.ascontiguousarray(
            generate_job_reqs(self.n_jobs, 
                              self.n_traits, 
                              self.n_clusters, 
                              self.ability_range), 
            dtype=np.float32
        )

        
        # Pre-allocate arrays
        self.current_ability = self.mirt._get_theta()
        self._match_qualities = np.zeros(self.n_jobs, dtype=np.float32)
        self._top_k_indices = np.zeros(self.K, dtype=np.int32)
        self._abs_diffs = np.zeros((self.n_jobs, self.n_traits), dtype=np.float32)


    def reset(self) -> np.ndarray:
        """Reset environment state."""
        self.mirt = AdaptiveMIRT(n_traits=self.n_traits)
        self.current_ability = self.mirt._get_theta()
        
        # Generate new job requirements

        # self.job_reqs = np.ascontiguousarray(
        #     np.random.uniform(
        #         self.ability_range[0],
        #         self.ability_range[1],
        #         size=(self.n_jobs, self.n_traits)
        #     ).astype(np.float32)
        # )
        self.job_reqs = np.ascontiguousarray(
            generate_job_reqs(self.n_jobs, 
                              self.n_traits, 
                              self.n_clusters, 
                              self.ability_range), 
            dtype=np.float32
        )
        return self.current_ability

    def _compute_reward(self, scores: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Optimized reward computation that maintains consistency with original version.
        Returns both reward and top-k indices.
        """
        # Get top-k indices using partition
        # top_k_indices = np.argpartition(scores, -self.K)[-self.K:]
        # np.copyto(self._top_k_indices, top_k_indices)
        
        # # Compute absolute differences
        # ability_expanded = np.expand_dims(self.current_ability, 0)
        # np.subtract(self.job_reqs, ability_expanded, out=self._abs_diffs)
        # np.abs(self._abs_diffs, out=self._abs_diffs)
        
        # # Compute mean differences along traits axis
        # np.mean(self._abs_diffs, axis=1, out=self._match_qualities)
        
        # # Convert to match qualities (1 - mean_diff)
        # # np.negative(self._match_qualities, out=self._match_qualities)
        # # np.multiply(self._match_qualities, 2.0, out=self._match_qualities)  # Scale factor
        # np.exp(self._match_qualities, out=self._match_qualities) #
        
        # # Compute reward for top-k jobs
        # selected_jobs = self.job_reqs[top_k_indices]
        # job_diversity = np.mean(np.std(selected_jobs, axis=0))
        
        # # Combine match quality with diversity bonus
        # base_reward = float(np.mean(self._match_qualities[top_k_indices]))
        # diversity_bonus = self.diversity_weight * job_diversity  # Scale the diversity bonus

        top_k_indices = np.argpartition(scores, -self.K)[-self.K:]
        np.copyto(self._top_k_indices, top_k_indices)
        
        # Compute absolute differences
        ability_expanded = np.expand_dims(self.current_ability, 0)
        np.subtract(self.job_reqs, ability_expanded, out=self._abs_diffs)
        np.abs(self._abs_diffs, out=self._abs_diffs)
        
        # Compute mean differences along traits axis
        np.mean(self._abs_diffs, axis=1, out=self._match_qualities)
        
        # Convert to match qualities (using negative exponential for better scaling)
        np.negative(self._match_qualities, out=self._match_qualities)
        np.exp(self._match_qualities, out=self._match_qualities)
        
        # Compute reward components
        selected_jobs = self.job_reqs[top_k_indices]
        job_diversity = np.mean(np.std(selected_jobs, axis=0))
        
        # Compute final reward - scale to [0,1] range for better stability
        base_reward = float(np.mean(self._match_qualities[top_k_indices]))
        total_reward = 2.0*base_reward + 1.0 * job_diversity
        # total_reward = total_reward / (1 + self.diversity_weight)  # Normalize to [0,1]
        
        return total_reward, self._top_k_indices

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take environment step."""
        # if len(action) != self.n_jobs:
        #     raise ValueError(f"Action must be length {self.n_jobs}, got {len(action)}")
        
        # # Ensure action is float32
        # if action.dtype != np.float32:
        #     action = action.astype(np.float32, copy=False)
        
        # # Compute reward and get top-k indices
        # reward, top_k_indices = self._compute_reward(action)
        
        # # Update MIRT state
        # _ = self.mirt.next_item()
        # resp = self.mirt.sim_resp()
        # self.current_ability = self.mirt._get_theta()
        
        # return self.current_ability, reward, False, {'top_k_jobs': top_k_indices}

        if len(action) != self.n_jobs:
            raise ValueError(f"Action must be length {self.n_jobs}, got {len(action)}")
        
        # Clip negative scores to 0 for safety
        action = np.maximum(action, 0)
        
        # Compute reward and get top-k indices
        reward, top_k_indices = self._compute_reward(action)
        
        # Update MIRT state
        _ = self.mirt.next_item()
        resp = self.mirt.sim_resp()
        self.current_ability = self.mirt._get_theta()
        
        return self.current_ability, reward, False, {'top_k_jobs': top_k_indices}

# class VocRecEnvOld:
#     """
#     Environment for job recommendation with MIRT-based ability estimation.
#     """
#     def __init__(
#         self, 
#         n_traits: int = 6,
#         n_jobs: int = 100,
#         top_k: int = 5,
#         ability_range: Tuple[float, float] = (-3, 3)
#     ):
#         self.n_traits = n_traits
#         self.n_jobs = n_jobs
#         self.K = top_k
#         self.ability_range = ability_range
#         self.mirt = AdaptiveMIRT(n_traits=n_traits)
#         self.job_reqs = np.random.uniform(ability_range[0], ability_range[1], 
#                                         size=(n_jobs, n_traits))
#         self.current_ability = None

#     def reset(self) -> np.ndarray:
#         self.mirt = AdaptiveMIRT(n_traits=self.n_traits)
#         self.current_ability = self.mirt._get_theta()
#         self.job_reqs = np.random.uniform(
#             self.ability_range[0],
#             self.ability_range[1],
#             size=(self.n_jobs, self.n_traits)
#         )
#         return self.current_ability

#     def _compute_reward(self, scores: np.ndarray) -> float:
#         top_k_indices = np.argsort(scores)[-self.K:]
#         rewards = []
#         for idx in top_k_indices:
#             job_req = self.job_reqs[idx]
#             match_quality = 1 - np.abs(self.current_ability - job_req).mean()
#             rewards.append(match_quality)
#         return np.mean(rewards)

#     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
#         if len(action) != self.n_jobs:
#             raise ValueError(f"Action must be length {self.n_jobs}, got {len(action)}")
            
#         reward = self._compute_reward(action)
        
#         next_item = self.mirt.next_item()
#         self.mirt.sim_resp()
#         self.mirt.update_theta()
        
#         self.current_ability = self.mirt._get_theta()
        
#         info = {
#             'top_k_jobs': np.argsort(action)[-self.K:]
#         }
        
#         return self.current_ability, reward, False, info

