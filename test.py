import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape:',env.observation_space.shape)
print('Number of actions:',env.action_space.n)

from dqn_agent import Agent



# 加载权重文件
Agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

state = env.reset()
for j in range(20000):
    action = Agent.act(state)
    env.render()
    state, reward, done, info = env.step(action)
    if done:
        state = env.reset()
env.close()