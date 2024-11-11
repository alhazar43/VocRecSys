# MDP definition

Tested definition:

- **State space**: $\mathcal{S} = [0,1] \text{ where } s_t = \alpha_t$ is latent ability estimation

- **Action space**:  $\mathcal{A} = \{r_c \in \mathbb{R}^+ \mid c \in \mathcal{C}\}$

- **Course space**:  $\mathcal{C}$ where each course $c$ has difficulty  $d_c \in [0,1]$

- **Top-K selection**:  $C_K(a_t) = \underset{S \subseteq \mathcal{C}, |S|=K}{\arg\max} \sum_{c \in S} r_c$

- **Reward**:  $R(s_t, a_t) = \frac{1}{K} \sum_{c \in C_K(a_t)} (1 - |\alpha_t - d_c|)$

- **Transition**:  $s_{t+1} \sim P(\cdot|s_t, a_t)$

- **Objective**:  $\max_{\pi} \mathbb{E} [\sum_{t=0}^{\infty} \gamma^t R(s_t, \pi(s_t))]$

- **Policy** $\pi_{\theta}(s_t) = \{r_c\}_{c \in \mathcal{C}}$

# DL defs
- **MSE Loss** $\mathcal{L}_{\text{MSE}}(\theta) = \mathbb{E}_{s_t \sim \mathcal{D}} \left[\frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} (r_c - (1-|\alpha_t - d_c|))^2\right]$
- **Contrastive Loss** $\mathcal{L}_{\text{cont}}(\theta) = \mathbb{E}_{s_t \sim \mathcal{D}} \left[-\log \frac{e^{r_c/\tau}}{e^{r_c/\tau} + \sum_{c' \in \mathcal{N}(c)} e^{r_{c'}/\tau}}\right]$
- **Combined Loss** $\mathcal{L}_{\text{total}}(\theta) = \lambda_1 \mathcal{L}_{\text{MSE}} + \lambda_2 \mathcal{L}_{\text{rank}} + \lambda_3 \mathcal{L}_{\text{cont}}$
- **Policy gradient Loss** $\mathcal{L}_{\text{PG}}(\theta) = -\mathbb{E}_{s_t \sim \mathcal{D}} \left[R(s_t, \pi_{\theta}(s_t)) \log \pi_{\theta}(s_t)\right]$

