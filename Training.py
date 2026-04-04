from Learning import Learning
import os
import gymnasium as gym
import ale_py
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
'''
这个文件是训练模块，是训练的主要部分
'''
n_envs=max(1,os.cpu_count()//2)#根据CPU核心数量设置环境数量，确保不会过度占用资源。

Train_env=make_atari_env('ALE/Tetris-v5',n_envs=n_envs,seed=0)
Train_env = VecFrameStack(Train_env, n_stack=4)

print('训练开始')
train_model=Learning(
    env=Train_env,
    tensorboard_log=r'D:\LLM\AIGaming\Train2\Train2_logs',
    model_path=r'D:\LLM\Models\Tetris_DQN_model_2.zip'
)
train_model.learn()
print('训练完成')