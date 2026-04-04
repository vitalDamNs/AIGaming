'''
这个文件是学习模块，包含了学习相关的函数和类。
'''
import gymnasium as gym 
from stable_baselines3 import DQN # 引入DQN算法,DQN是用来解决强化学习问题的
import ale_py


class Learning:
    '''
    学习类，包含了学习相关的函数和属性。
    之后进行超参数微调时，可以在这里进行修改。
    '''
    def __init__(self,Policy='CnnPolicy',
                 env=None,
                 learning_rate=1e-4,
                 buffer_size=100000,
                 learning_starts=50000,
                 batch_size=32,tau=1.0,
                 gamma=0.99,
                 train_freq=4,
                 gradient_steps=1,
                 target_update_interval=1000,
                 exploration_fraction=0.1,
                 exploration_final_eps=0.05,
                 verbose=1,
                 tensorboard_log='',
                 device='cuda',
                 total_timesteps=1000000,
                 model_path=''
                 ):

        '''
        初始化学习类
        Policy：策略网络，用于选择动作。
        env：环境，用于训练智能体。
        learning_rate：学习率，用于更新策略网络。
        buffer_size：经验回放缓冲区大小，用于存储经验样本。
        learning_starts：学习开始的步数，用于等待经验样本积累到一定数量后再开始学习。
        batch_size：批量大小，用于更新策略网络。
        tau：软更新参数，用于更新目标网络的参数，防止网络参数更新过快导致的不稳定性。
        gamma：折扣因子，用于计算未来奖励的折扣。
        train_freq：训练频率，每训练4步进行一次更新。
        gradient_steps：梯度步长，用于更新网络参数的步长。
        target_update_interval：目标网络更新间隔，每1000步更新一次目标网络的参数。
        exploration_fraction：探索分数，用于控制探索和利用的平衡,探索率在总步数的前10%内从1降到0.1。
        exploration_final_eps：最终探索率，探索率在总步数的最后10%内从0.05降到0.01。
        verbose：是否打印学习进度。
        tensorboard_log：是否记录学习过程。
        device：是否使用cuda。
        total_timesteps：总步数，用于训练智能体。
        model_path：模型路径，用于保存模型。
        '''
        self.Policy=Policy
        self.env=env
        self.learning_rate=learning_rate
        self.buffer_size=buffer_size
        self.learning_starts=learning_starts
        self.batch_size=batch_size
        self.tau=tau
        self.gamma=gamma
        self.train_freq=train_freq
        self.gradient_steps=gradient_steps
        self.target_update_interval=target_update_interval
        self.exploration_fraction=exploration_fraction
        self.exploration_final_eps=exploration_final_eps
        self.verbose=verbose
        self.tensorboard_log=tensorboard_log
        self.device=device
        self.total_timesteps=total_timesteps
        self.model_path=model_path


    def learn(self):
        '''
        学习函数，根据设置的算法进行学习。
        此处运用DQN算法进行学习。
        '''
        model=DQN(
            policy=self.Policy,
            env=self.env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,   
            batch_size=self.batch_size,
            tau=self.tau,                  
            gamma=self.gamma,                 
            train_freq=self.train_freq,            
            gradient_steps=self.gradient_steps,
            target_update_interval=self.target_update_interval,
            exploration_fraction=self.exploration_fraction,
            exploration_final_eps=self.exploration_final_eps, 
            verbose=self.verbose,
            tensorboard_log=self.tensorboard_log
        )
        model.learn(total_timesteps=self.total_timesteps)
        model.save(self.model_path)
        return "学习完成,模型已保存在"+self.model_path



'''        
这一段是用来测试学习模块的功能，主要是判断是否能够在cuda环境下正常运行，并且能够正确地使用DQN算法进行学习。
在这个测试中，我们创建了一个Taxi-v3环境，并使用DQN算法进行学习。
'''
# if __name__=='__main__': 
#     env=gym.make('Taxi-v3', render_mode="human")
#     learning_model=Learning(env,algorithm='DQN',verbose=1,total_timesteps=10000,cnnPolicy=False,device='cuda')
#     learning_model.learn()
#     env.close()
