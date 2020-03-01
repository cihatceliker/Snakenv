import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import sys


class Brain(nn.Module):

    def __init__(self, *args):
        super(Brain, self).__init__()
        layers = []
        for i in range(1, len(args)):
            layers.append(nn.Linear(args[i-1], args[i]))
            if i != len(args)-1: layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda")
        self.to(self.device)

    def forward(self, state):
        x = self.layers(state)
        return x

    def mutate(self, mutation_rate):
        dc = self.state_dict()
        for data in dc:
            mask = torch.rand_like(dc[data]) < mutation_rate
            sz = len(dc[data][mask])
            dc[data][mask] = torch.randn(sz, device=self.device)


class Agent():
    
    def __init__(self, Q, target_Q, num_actions, eps_start=1.0, eps_end=0.01,
                 eps_decay=0.995, gamma=0.99, alpha=5e-4, batch_size=64, memory_capacity=10000, tau=1e-3):
        self.Q = Q
        self.Q.optimizer = optim.Adam(self.Q.parameters(), lr=alpha)
        self.target_Q = target_Q
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(memory_capacity)

    def store_experience(self, *args):
        self.replay_memory.push(args)

    def select_action(self, state):
        state = torch.Tensor(state).to(self.Q.device)
        sample = np.random.random()
        if sample > self.eps_start:
            self.Q.eval()
            with torch.no_grad():
                action = torch.argmax(self.Q(state)).item()
            self.Q.train()
        else:
            action = np.random.randint(self.num_actions)
        actions = [0] * self.num_actions
        actions[action] = 1
        return actions

    def learn(self):
        if len(self.replay_memory.memory) < self.batch_size:
            return
        batch = self.replay_memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            [[None for _ in range(self.batch_size)] for _ in range(5)]

        for i in range(self.batch_size):
            state_batch[i], action_batch[i], reward_batch[i], next_state_batch[i], done_batch[i] = batch[i][0]

        state_batch = torch.tensor(state_batch, device=self.Q.device)
        action_batch = torch.tensor(action_batch, dtype=torch.float, device=self.Q.device)
        reward_batch = torch.tensor(reward_batch, device=self.Q.device)
        next_state_batch = torch.tensor(next_state_batch, device=self.Q.device)
        done_batch = torch.tensor(done_batch, device=self.Q.device)

        action_values = torch.arange(self.num_actions, dtype=torch.float, device=self.Q.device)
        action_indices = action_batch.matmul(action_values).long()
        batch_index = torch.arange(self.batch_size).long()

        output = self.Q(state_batch)
        q_target = output.clone()
        q_target[batch_index, action_indices] = reward_batch + \
            self.gamma * torch.max(self.target_Q(next_state_batch), dim=1)[0] * done_batch

        loss = self.Q.loss(output, q_target).to(self.Q.device)
        self.Q.optimizer.zero_grad()
        loss.backward()
        #for param in self.Q.parameters(): param.grad.data.clamp_(-1, 1)
        self.Q.optimizer.step()

        for target_param, local_param in zip(self.target_Q.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, size):
        return random.sample(self.memory, size)
