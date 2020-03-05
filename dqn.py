import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math
import sys

device = torch.device("cuda")
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#torch.backends.cudnn.benchmark = True


class Brain(nn.Module):

    def __init__(self, in_size, out_size, fc1_size=64, fc2_size=64):
        super(Brain, self).__init__()
        self.fc1 = nn.Linear(in_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.out = nn.Linear(fc2_size, out_size)
        self.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


class DuelingDQNBrain(nn.Module):

    def __init__(self, in_size, out_size, fc1_size=64, fc2_size=32):
        super(DuelingDQNBrain, self).__init__()
        self.in_size = in_size
        self.fc1 = nn.Linear(in_size, fc1_size)
        self.fc_val = nn.Linear(fc1_size, fc2_size)
        self.fc_adv = nn.Linear(fc1_size, fc2_size)
        self.val = nn.Linear(fc2_size, 1)
        self.adv = nn.Linear(fc2_size, out_size)
        self.to(device)

    def forward(self, state):
        # !!!
        if len(state) == self.in_size:
            state = state.view(-1, self.in_size)
        x = torch.relu(self.fc1(state))
        val = torch.relu(self.fc_val(x))
        adv = torch.relu(self.fc_adv(x))
        val = self.val(val)
        adv = self.adv(adv)
        adv_mean = torch.mean(adv, dim=1, keepdim=True)
        x = val + adv - adv_mean
        return x


class Agent():
    
    def __init__(self, local_Q, target_Q, num_actions, eps_start=1.0, eps_end=0.01,
                 eps_decay=0.995, gamma=0.99, alpha=5e-4, batch_size=128, memory_capacity=10000, tau=1e-3):
        self.local_Q = local_Q
        self.target_Q = target_Q
        self.target_Q.load_state_dict(self.local_Q.state_dict())
        self.target_Q.eval()
        self.optimizer = optim.Adam(self.local_Q.parameters(), lr=alpha)
        #self.loss = nn.MSELoss()
        self.loss = nn.SmoothL1Loss()
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
        self.batch_index = np.arange(self.batch_size)

    def store_experience(self, *args):
        self.replay_memory.push(args)

    def select_action(self, state):
        state = torch.Tensor(state).to(device)
        sample = np.random.random()
        if sample > self.eps_start:
            with torch.no_grad():
                action = torch.argmax(self.local_Q(state)).item()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def learn(self):
        if len(self.replay_memory.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_memory.sample(self.batch_size)

        output = self.local_Q(state_batch)
        target = output.clone()

        # vanilla dqn
        target[self.batch_index, action_batch] = torch.max(self.target_Q(next_state_batch), dim=1)[0]
        
        # double dqn
        #target[self.batch_index, action_batch] = \
        # self.target_Q(next_state_batch)[self.batch_index, torch.argmax(self.local_Q(next_state_batch), dim=1)]

        target[self.batch_index, action_batch] *= self.gamma * done_batch
        target[self.batch_index, action_batch] += reward_batch

        loss = self.loss(output, target.detach()).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update
        for target_param, local_param in zip(self.target_Q.parameters(), self.local_Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)


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

        batch = [*zip(*batch)]
        
        state_batch = torch.tensor(batch[0], device=device)
        action_batch = torch.tensor(batch[1], device=device)
        reward_batch = torch.tensor(batch[2], device=device)
        next_state_batch = torch.tensor(batch[3], device=device)
        done_batch = torch.tensor(batch[4], device=device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch