import gymnasium as gym
import torch
import ale_py

'''
这一段是用来是检查torch环境，包括是否支持cuda。
'''
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())

'''
这一段是用来检查gym环境，特别是ALE环境是否正确注册和可用。
'''
gym.register_envs(ale_py)
env=gym.make('ALE/Tetris-v5', render_mode='human')
obs,info=env.reset()
episode_over=False

while not episode_over:
    action=env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()
