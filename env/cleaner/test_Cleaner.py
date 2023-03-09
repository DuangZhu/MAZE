import random

import gym
from stable_baselines3.common.env_checker import check_env

from envCleaner import EnvCleaner, EnvCleaner_onehot , EnvCleaner_oneimage

if __name__ == '__main__':
    env = EnvCleaner_oneimage({"map_size":7,"seed":0,"N_agent":5,"partical_obs":2})
    check_env(env)
    
    max_iter = 500
    num_env = 5
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # 实验并行计算
    env_ = lambda: env
    envs = gym.vector.SyncVectorEnv([env_ for i in range(num_env)])
    
    
    envs.reset()
    for i in range(max_iter):
        print("iter= ", i)
        #env.render()
        action_list = {}
        action_list = [env.action_space.sample() for _ in range(num_env)]
        obs, reward, done, info = envs.step(action_list) #注意这里的reward是集体的
        print(done)
        print('reward', reward)