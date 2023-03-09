import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from envCleaner import EnvCleaner
ray.init()
t = tune.run(PPOTrainer, config={
   "env": 'gym_cleaner',
   "framework": "torch",
   "log_level": "INFO",
   
},stop={
        'episode_reward_max':91
    })
# import gym
# env = gym.make('gym_cleaner:cleaner-v0')
# env.reset()
# for _ in range(1000):
#     action = env.action_space.sample()
#     print(action)
#     env.step(action) # take a random action
# env.close() # 關閉視圖
