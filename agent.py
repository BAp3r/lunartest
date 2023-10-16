import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim,lr=0.001,batch_size=128):
        self.batch_size=batch_size
        self.lr=lr
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, input_dim, output_dim, lr=0.001):
        self.dqn = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def choose_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([i for i in range(self.dqn.fc3.out_features)])
        else:
            state = torch.tensor([state], dtype=torch.float32)
            actions = self.dqn(state)
            return torch.argmax(actions).item()

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor([state], dtype=torch.float32)
        next_state = torch.tensor([next_state], dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        #引入额外的奖励
        distance_to_center_before = abs(state[0][0].item())
        distance_to_center_after = abs(next_state[0][0].item())
        if distance_to_center_after < distance_to_center_before:
            reward += 3.0
        #引入姿态奖励
        angle_before = state[0][4].item()
        angle_after = next_state[0][4].item()
        if abs(angle_after) < abs(angle_before):
            reward += 2.0
        #引入速度奖励
        velocity_before = state[0][3].item()
        velocity_after = next_state[0][3].item()
        if abs(velocity_after) < abs(velocity_before):
            reward += 1.0
            
        if done:
            target = reward
        else:
            target = reward + 0.99 * torch.max(self.dqn(next_state))

        prediction = self.dqn(state)[0][action]
        loss = self.criterion(prediction, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
