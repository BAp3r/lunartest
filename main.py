import gym
import numpy as np
from agent import Agent
import torch
import matplotlib.pyplot as plt

def calculate_bonus_reward(state, next_state):
    distance_to_center_before = abs(state[0])
    distance_to_center_after = abs(next_state[0])
    
    if distance_to_center_after < distance_to_center_before:
        return 1.0
    else:
        return 0.0

env = gym.make('LunarLander-v2')
agent = Agent(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n)

n_episodes = 1000
epsilon = 1.0
epsilon_decay = 0.995
step_count = 0

for episode in range(n_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()  # 现在这行应该能正常工作
        action = agent.choose_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)  # 使用环境的step函数
        
        bonus_reward = calculate_bonus_reward(state, next_state)
        modified_reward = reward + bonus_reward
        
        agent.train(state, action, modified_reward, next_state, done)
        state = next_state
        total_reward += modified_reward

        step_count += 1
        if step_count % 500 == 0:
            #torch.save(agent.dqn.state_dict(), f"model_step_{step_count}.pth")
            torch.save(agent.dqn.state_dict(), "model.pth")
    epsilon *= epsilon_decay
    print(f"Episode: {episode}, Total Reward: {total_reward}")

env.close()


# 使用matplotlib绘制奖励曲线
plt.plot(reward)
plt.ylabel('Total rewards')
plt.xlabel('Episodes')
plt.show()
