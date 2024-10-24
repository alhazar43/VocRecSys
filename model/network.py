import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, input_dim, n_courses):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, n_courses)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  
        return logits


class CriticNet(nn.Module):
    def __init__(self, input_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value
