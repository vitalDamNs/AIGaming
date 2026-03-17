import gymnasium as gym
import torch

'''
This file is used to test the environment, and make sure it can be rendered properly.
'''

# env=gym.make('ALE/Tetris-v5', render_mode='human')
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
