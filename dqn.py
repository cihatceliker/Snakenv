import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import sys


device = torch.device("cuda")

class Brain(nn.Module):

    def __init__(self, *args):
        super(Brain, self).__init__()
        layers = []
        for i in range(1, len(args)):
            layers.append(nn.Linear(args[i-1], args[i]))
            if i != len(args)-1: layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.to(device)

    def forward(self, state):
        x = self.layers(state)
        return x

    def mutate(self, mutation_rate):
        dc = self.state_dict()
        for data in dc:
            mask = torch.rand_like(dc[data]) < mutation_rate
            sz = len(dc[data][mask])
            dc[data][mask] = torch.randn(sz, device=device)


class Agent():
    
    def __init__(self, local_Q, target_Q, num_actions, eps_start=1.0, eps_end=0.01,
                 eps_decay=0.995, gamma=0.99, alpha=5e-4, batch_size=64, memory_capacity=10000, tau=1e-3):
        self.local_Q = local_Q
        self.target_Q = target_Q
        self.target_Q.load_state_dict(self.local_Q.state_dict())
        self.optimizer = optim.Adam(self.local_Q.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(memory_capacity)
        self.scores = []
        self.episodes = []

    def store_experience(self, *args):
        self.replay_memory.push(args)

    def select_action(self, state):
        state = torch.Tensor(state).to(device)
        sample = np.random.random()
        if sample > self.eps_start:
            self.local_Q.eval()
            with torch.no_grad():
                action = torch.argmax(self.local_Q(state)).item()
            self.local_Q.train()
        else:
            action = np.random.randint(self.num_actions)
        actions = [0.] * self.num_actions
        actions[action] = 1
        return actions

    def learn(self):
        if len(self.replay_memory.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_memory.sample(self.batch_size)

        action_values = torch.arange(self.num_actions, dtype=torch.float, device=device)
        action_indices = action_batch.matmul(action_values).long()
        batch_index = torch.arange(self.batch_size).long()

        output = self.local_Q(state_batch)
        target = output.clone()

        target[batch_index, action_indices] = reward_batch + \
            self.gamma * torch.max(self.target_Q(next_state_batch), dim=1)[0] * done_batch
    
        loss = self.loss(output, target).to(device)

        self.optimizer.zero_grad()
        loss.backward()
        
        #for param in self.local_Q.parameters(): param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        # soft update
        for target_param, local_param in zip(self.target_Q.parameters(), self.local_Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args[0]
        self.position = (self.position + 1) % self.capacity

    def sample(self, size):
        batch = random.sample(self.memory, size)

        batch = tuple([*zip(*batch)])

        state_batch = torch.tensor(batch[0], device=device)
        action_batch = torch.tensor(batch[1], device=device)
        reward_batch = torch.tensor(batch[2], device=device)
        next_state_batch = torch.tensor(batch[3], device=device)
        done_batch = torch.tensor(batch[4], device=device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch