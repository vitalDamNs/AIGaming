'''
这个文件是学习模块，包含了学习相关的函数和类。
'''
import gymnasium as gym 
from stable_baselines3 import DQN # 引入DQN算法,DQN是用来解决强化学习问题的

class Learning:
    '''
    学习类，包含了学习相关的函数和属性。
    '''
    def __init__(self,env,algorithm='DQN',verbose=0,total_timesteps=10,cnnPolicy=True,device='cuda'):
        '''
        初始化学习类，设置环境、算法和是否打印日志。
        '''
        self.algorithm=algorithm
        self.verbose=verbose
        self.total_timesteps=total_timesteps
        self.env=env
        self.algorithm=algorithm
        self.cnnPolicy=cnnPolicy
        self.device=device

    def learn(self):
        '''
        学习函数，根据设置的算法进行学习。
        如果是DQN算法，根据是否使用卷积神经网络策略网络，选择不同的模型。
        cnnPolicy=True表示使用卷积神经网络策略网络，适用于处理图像输入的环境。
        cnnPolicy=False表示使用全连接层策略网络，适用于处理非图像输入的环境。
        '''
        
        if self.algorithm=='DQN':
            if self.cnnPolicy:
                model=DQN('CnnPolicy',self.env,verbose=self.verbose,device=self.device)
                print(model.policy.device)
            else:
                model=DQN('MlpPolicy',self.env,verbose=self.verbose,device=self.device)
                print(model.policy.device)
            model.learn(total_timesteps=self.total_timesteps)

'''
这一段是用来测试学习模块的功能，主要是判断是否能够在cuda环境下正常运行，并且能够正确地使用DQN算法进行学习。
在这个测试中，我们创建了一个Taxi-v3环境，并使用DQN算法进行学习。
'''
if __name__=='__main__': 
    env=gym.make('Taxi-v3', render_mode="human")
    learning_model=Learning(env,algorithm='DQN',verbose=1,total_timesteps=10000,cnnPolicy=False,device='cuda')
    learning_model.learn()
    env.close()
