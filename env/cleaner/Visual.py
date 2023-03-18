# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from envCleaner import EnvCleaner, EnvCleaner_onehot, EnvCleaner_oneimage
from gym import logger, spaces
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from ppo_cleaner_lstm import Agent


env = EnvCleaner_oneimage({"map_size":9,"seed":5,"N_agent":1,"partical_obs":3,"start_position":[[1,1]]})
agent_id_pos = torch.zeros(env.N_agent,3)
for i in range(env.N_agent):
    agent_id_pos[i] = torch.tensor([i,env.start_position[i%len(env.start_position)][0],
                                env.start_position[i%len(env.start_position)][1]])
agent_id_pos = agent_id_pos.repeat(1,1).view(1,env.N_agent,3)    
agent = Agent(env)
next_obs = torch.unsqueeze(torch.Tensor(env.reset()),dim=0)
next_done = torch.zeros(1)
agent.load_state_dict(torch.load("D:/00MYCODE/Project/Maze/env/cleaner/runs/ippo/single_/27000000_params.pth"))
next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size),
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size),)

for i in range(50000):
    env.render()
    action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done, agent_id_pos= agent_id_pos)
    next_obs, reward, next_done, info = env.step(action.numpy())
    next_done = torch.Tensor([next_done])
    next_obs = torch.unsqueeze(torch.Tensor(next_obs) ,dim=0)
    agent_id_pos = torch.Tensor(info["agent_info"])
    if next_done == True:
        next_obs = torch.unsqueeze(torch.Tensor(env.reset()),dim=0)
        next_done = torch.zeros(1)
