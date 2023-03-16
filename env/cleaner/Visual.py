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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, 5), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)
        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std) #初始权重正交化
    torch.nn.init.constant_(layer.bias, bias_const) #初始bias，常数初始化
    return layer


env = EnvCleaner_oneimage({"map_size":9,"seed":3,"N_agent":1,"partical_obs":3,"start_position":[[1,1]]})
agent = Agent(env)
next_obs = torch.unsqueeze(torch.Tensor(env.reset()),dim=0)
next_done = torch.zeros(1)
agent.load_state_dict(torch.load("./env/cleaner/runs/default+4env+24env_num+3kw_step/18000000_params.pth"))
next_lstm_state = (
        torch.ones(agent.lstm.num_layers, 1, agent.lstm.hidden_size),
        torch.ones(agent.lstm.num_layers, 1, agent.lstm.hidden_size),)

for i in range(5000):
    action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
    next_obs, reward, next_done, info = env.step(action.numpy())
    next_done = torch.Tensor([next_done])
    next_obs = torch.unsqueeze(torch.Tensor(next_obs) ,dim=0)
    print(action)
    env.render()
