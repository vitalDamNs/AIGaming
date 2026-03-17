import gymnasium as gym
import os
'''
use 'gymnasium.envs.registry.keys()' to check all available environments,
and put them in a text file for later use. 
'''
filepath='Envs.txt'
if os.path.exists(filepath):
    env_list=[]
    with open(filepath,'r+') as f:
        env_list=f.read().splitlines()
        new_env=list(gym.envs.registry.keys())
        for env in new_env:
            if env not in env_list:
                f.write(env+'\n')
else:        
    with open(filepath,'w') as f:
        for env in gym.envs.registry.keys():
            f.write(env+'\n')