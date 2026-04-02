import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
import time

'''
这个文件是用来测试模型的训练情况的
'''
model_path=r'D:\LLM\Models\Tetris_DQN_model.zip'
model=DQN.load(model_path)

env=gym.make('ALE/Tetris-v5', render_mode="human")
obs,info=env.reset()

while True:
    action,_=model.predict(obs,deterministic=True)#deterministic=True表示模型选择最优动作，而非随机探索
    obs,reward,terminated,truncated,info=env.step(action)
    time.sleep(0.02)
    if terminated or truncated:
        obs,info=env.reset()
   