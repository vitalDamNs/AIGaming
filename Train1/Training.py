from Learning import Learning
import gymnasium as gym
import ale_py
'''
这个文件是训练模块，是训练的主要部分
'''
Train_env=gym.make('ALE/Teris-v5', render_mode="human")
print('训练开始')
train_model=Learning(Train_env,algorithm='DQN',verbose=1,total_timesteps=100000,cnnPolicy=False,device='cuda',Game_name='Tetris')
train_model.learn()
print('训练完成')