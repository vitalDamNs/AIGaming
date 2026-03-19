import gymnasium as gym
import torch
import ale_py

'''
这一段是用来是检查torch环境，包括是否支持cuda。
'''
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

'''
这一段是用来检查gym环境，特别是ALE环境是否正确注册和可用。
'''
gym.register_envs(ale_py)
# env=gym.make('ALE/Tetris-v5', render_mode='human')
env = gym.make('ALE/Tetris-v5', continuous=True, render_mode="human")
#continuous=True表示环境将持续运行，直到游戏结束，而不是在每一步后重置。

obs,info=env.reset()
episode_over=False

while not episode_over:
    '''
    action=env.action_space.sample()是随机采样一个动作，这个动作是从环境的动作空间中随机选择的。
    这个动作将被传递给env.step(action)函数，以便环境根据这个动作更新状态并返回新的观察、奖励、是否结束等信息。
    terminated和truncated是两个布尔值，分别表示游戏是否因为达到终止条件（如游戏结束）而结束，或者因为达到截断条件（如时间限制）而结束。
    episode_over是一个布尔值，当游戏结束或截断时为True，循环将停止。
    '''
    action=env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()
