import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_jobs):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_jobs)

    def forward(self, observation):
        # Ensure input is on the same device as network parameters
        if observation.device != self.fc1.weight.device:
            observation = observation.to(self.fc1.weight.device)
            
        x = observation.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, observation):
        # Ensure input is on the same device as network parameters
        if observation.device != self.fc1.weight.device:
            observation = observation.to(self.fc1.weight.device)
            
        x = observation.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value

# class ActorNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, n_jobs):
#         super(ActorNet, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, n_jobs)

#     def forward(self, observation):
#         # Flatten observation to match input_dim
#         x = observation.flatten(start_dim=1)  # Flatten starting from the batch dimension
#         x = F.relu(self.fc1(x))
#         logits = self.fc2(x)
#         return logits

# class CriticNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(CriticNet, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)

#     def forward(self, observation):
#         # Flatten observation to match input_dim
#         x = observation.flatten(start_dim=1)  # Flatten starting from the batch dimension
#         x = F.relu(self.fc1(x))
#         value = self.fc2(x)
#         return value