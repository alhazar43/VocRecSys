import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value):
        # Handle single input case by adding batch dimension
        if len(query.shape) == 2:
            query = query.unsqueeze(0)
        if len(key.shape) == 2:
            key = key.unsqueeze(0)
        if len(value.shape) == 2:
            value = value.unsqueeze(0)
            
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Linear projections
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        
        # Split into heads
        q = q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # Combine heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len_q, self.d_model)
        
        return self.out(output)

class Actor(nn.Module):
    """Policy network that directly outputs job scores"""
    def __init__(self, n_traits: int, n_jobs: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_traits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_jobs),
            nn.Softplus()  # Ensure positive scores
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Directly output job scores"""
        return self.network(state)

class Critic(nn.Module):
    """Value network"""
    def __init__(self, n_traits: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_traits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
    

class ActorCritic(nn.Module):
    def __init__(self, n_traits: int, n_jobs: int, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        
        self.trait_embedding = nn.Sequential(
            nn.Linear(n_traits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.job_embedding = nn.Sequential(
            nn.Linear(n_traits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_jobs),
            nn.ReLU()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Handle single state input
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # [1, n_traits]
        batch_size = state.size(0)
        
        # Get and prepare job requirements
        job_reqs = torch.FloatTensor(self.env.job_reqs).to(state.device)  # [n_jobs, n_traits]
        job_reqs = job_reqs.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_jobs, n_traits]
        
        # Embeddings
        state_emb = self.trait_embedding(state)  # [batch_size, hidden_dim]
        state_emb = state_emb.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        job_emb = self.job_embedding(job_reqs)  # [batch_size, n_jobs, hidden_dim]
        
        # Attention
        attended = self.attention(state_emb, job_emb, job_emb)  # [batch_size, 1, hidden_dim]
        attended = self.norm1(attended)
        
        # Combine features
        features = torch.cat([
            attended.squeeze(1),  # [batch_size, hidden_dim]
            state_emb.squeeze(1)  # [batch_size, hidden_dim]
        ], dim=-1)  # [batch_size, hidden_dim*2]
        features = self.ffn(features)  # [batch_size, hidden_dim]
        
        # Actor head: generates scores for each job
        job_scores = self.actor(features)  # [batch_size, n_jobs]
        action_probs = F.softmax(job_scores, dim=-1)  # [batch_size, n_jobs]
        
        # Critic head: generates state value
        value = self.critic(features)  # [batch_size, 1]
        
        return job_scores, action_probs, value
    
    def set_env(self, env):
        self.env = env