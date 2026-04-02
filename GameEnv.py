import os
import multiprocessing
import gymnasium as gym
import torch
import ale_py
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
'''
这一段是用来是检查torch环境，包括是否支持cuda。
'''
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

'''
这一段是用来检查gym环境，特别是ALE环境是否正确注册和可用。
'''
# gym.register_envs(ale_py)
# # env=gym.make('ALE/Tetris-v5', render_mode='human')
# env = make_atari_env('ALE/Tetris-v5', n_envs=1, seed=0)
# env = VecFrameStack(env, n_stack=4)


# obs,info=env.reset()
# print('初始化环境：',info)
# '''
# 这里是为了查看奖励值是否正常，正常情况下奖励值应该增加，直到游戏结束或截断。
# '''
# for _ in range(100):
#     action=env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(reward)
#     if terminated or truncated:
#         obs,info=env.reset()

# episode_over=False

# while not episode_over:
#     '''
#     action=env.action_space.sample()是随机采样一个动作，这个动作是从环境的动作空间中随机选择的。
#     这个动作将被传递给env.step(action)函数，以便环境根据这个动作更新状态并返回新的观察、奖励、是否结束等信息。
#     terminated和truncated是两个布尔值，分别表示游戏是否因为达到终止条件（如游戏结束）而结束，或者因为达到截断条件（如时间限制）而结束。
#     episode_over是一个布尔值，当游戏结束或截断时为True，循环将停止。
#     '''
#     action=env.action_space.sample()
#     obs, reward, terminated, truncated, _ = env.step(action)
#     print('奖励值：',reward)
#     episode_over = terminated or truncated

# env.close()

'''
这一段是用来检查电脑的硬件环境，例如cpu核心数量等。
'''
print('CPU核心数量：',os.cpu_count())
print('多进程数量：',multiprocessing.cpu_count())
