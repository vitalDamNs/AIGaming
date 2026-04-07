import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
import time
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, WarpFrame
)
from gymnasium.wrappers import FrameStackObservation
import numpy as np

'''
这是一个自定义的观察包装器，用于将环境的观察空间从(4,84,84,1) 转换为 (4,84,84)。减少输入维度。
'''
class SqueezeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape=self.observation_space.shape
        new_shape=old_shape[:-1]#去掉最后一个维度
        self.observation_space=gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8
        )

    def observation(self, obs):
        return obs.squeeze(-1)
    
'''
以下是用来测试模型的训练情况的
'''
def test_env():
    env=gym.make('ALE/Tetris-v5', render_mode='human')
    env=NoopResetEnv(env, noop_max=30)#在环境重置时执行随机数量的No-op动作，增加环境的随机性。
    env=MaxAndSkipEnv(env, skip=4)#在每个动作之间跳过4帧，减少环境的频率，提高训练效率。
    env=EpisodicLifeEnv(env)#将环境转换为结束时奖励为0的环境，避免奖励累计。
    env=WarpFrame(env)#将环境的图像帧转换为84x84的灰度图像，减少输入维度。
    env=FrameStackObservation(env, stack_size=4)#将连续的4帧图像堆叠在一起，提供更多的环境信息。
    env=SqueezeObservation(env)#将环境的观察空间从(4,84,84,1) 转换为 (4,84,84)。减少输入维度。
    return env

model_path=r'D:\LLM\Models\Tetris_DQN_model_2.zip'
model=DQN.load(model_path)

env=test_env()
obs,info=env.reset()
total_reward=0

print('测试开始')
while True:
    action, _ = model.predict(obs, deterministic=True)  # deterministic=True表示模型选择最优动作，而非随机探索
    obs,reward,terminated,truncated,info =env.step(action)
    total_reward+=reward

    time.sleep(0.02)

    if terminated or truncated:
        print(f'游戏结束，总得分: {total_reward}')
        obs,info=env.reset()
        total_reward=0

