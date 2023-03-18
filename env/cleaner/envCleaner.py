import copy
import random

import cv2
import gym
import numpy as np
import torch
from gym import logger, spaces
from gym.utils import seeding

import maze


class EnvCleaner_onehot(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,env_config):
        
        self.map_size = env_config["map_size"]
        self.seed = env_config["seed"]
        self.occupancy = self.generate_maze(self.seed)
        self.N_agent = env_config["N_agent"]
        self.agt_pos_list = []
        self.partical_obs = env_config["partical_obs"]
        for i in range(self.N_agent):
            self.agt_pos_list.append([1, 1])
        self.np_random = env_config["seed"]
        self.action_space = spaces.Tuple([spaces.Discrete(4) for _ in range(self.N_agent)])
        # self.state_space = spaces.Box(low=-1, high=1, shape=(state_size,), dtype="float32")
        # obs shape = id+pos+(self.partical_obs*2+1)*(self.partical_obs*2+1)
        self.observation_space = spaces.Tuple([spaces.Box(low=0, high=1, shape=(1,(self.partical_obs*2+1)*(self.partical_obs*2+1)+3), dtype=np.float64) for _ in range(self.N_agent)])
        self.start_position = [[1,1]]
        
        
        
    def generate_maze(self, seed):
        symbols = {
            # default symbols
            'start': 'S',
            'end': 'X',
            'wall_v': '|',
            'wall_h': '-',
            'wall_c': '+',
            'head': '#',
            'tail': 'o',
            'empty': ' '
        }
        #生成地图的关键，改变地图形式可以参考第一句
        maze_obj = maze.Maze(int((self.map_size - 1) / 2), int((self.map_size - 1) / 2), seed, symbols, 1)
        grid_map = maze_obj.to_np()
        
        for i in range(self.map_size):
            for j in range(self.map_size):
                if grid_map[i][j] == 0:
                    grid_map[i][j] = 2
        return grid_map

    def _step(self, action_list):
        reward = 0
        for i in range(len(action_list)):
            if action_list[i] == 0:     # up
                if self.occupancy[self.agt_pos_list[i][0] - 1][self.agt_pos_list[i][1]] != 1:  # if can move
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] - 1
            if action_list[i] == 1:     # down
                if self.occupancy[self.agt_pos_list[i][0] + 1][self.agt_pos_list[i][1]] != 1:  # if can move
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] + 1
            if action_list[i] == 2:     # left
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] - 1] != 1:  # if can move
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] - 1
            if action_list[i] == 3:     # right
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] + 1] != 1:  # if can move
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] + 1
            if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] == 2:   # if the spot is dirty
                self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] = 0
                reward = reward + 1
        return reward
    
    def step(self, action_list):
        done = False
        #Calculate individual reward
        self.in_reward = []
        total_reward = 0
        for i in range(len(action_list)):
            if action_list[i] == 0:     # up
                if self.occupancy[self.agt_pos_list[i][0] - 1][self.agt_pos_list[i][1]] != 1:  # if can move
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] - 1
            if action_list[i] == 1:     # down
                if self.occupancy[self.agt_pos_list[i][0] + 1][self.agt_pos_list[i][1]] != 1:  # if can move
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] + 1
            if action_list[i] == 2:     # left
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] - 1] != 1:  # if can move
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] - 1
            if action_list[i] == 3:     # right
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] + 1] != 1:  # if can move
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] + 1
            if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] == 2:   # if the spot is dirty
                self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] = 0
                self.in_reward.append(1.)
                total_reward += 1
            else:
                self.in_reward.append(0.)
        # Calculate pensonal observation with three channel
        globel_obs = self.get_global_obs()
        if self.partical_obs != 1:
            map = np.ones([self.map_size + self.partical_obs*2-2,self.map_size + self.partical_obs*2-2,3])
            map[self.partical_obs-1:-self.partical_obs+1,self.partical_obs-1:-self.partical_obs+1,:] = globel_obs
        else:
            map = globel_obs
        map = (map[:,:,0]*0.5+map[:,:,1]*0.3+map[:,:,2]*0.2).squeeze()
        obs = []
        for i in range(len(action_list)):
                per_obs = np.concatenate((np.array([i/self.N_agent,self.agt_pos_list[i][0]/self.map_size,self.agt_pos_list[i][1]/self.map_size]),map[self.agt_pos_list[i][0]-1:self.agt_pos_list[i][0]+2*self.partical_obs,self.agt_pos_list[i][1]-1:self.agt_pos_list[i][1]+2*self.partical_obs].ravel()))
                obs.append(per_obs[np.newaxis,:])

        # Calculate done
        if 2 not in self.occupancy:
            done = True
        info = {}
        
        
        return obs, total_reward, done, info

    def get_global_obs(self):
        obs = np.zeros((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.occupancy[i, j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                if self.occupancy[i, j] == 2:
                    obs[i, j, 0] = 0.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 0.0
        for i in range(self.N_agent):
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 0] = 1.0
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 1] = 0.0
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 2] = 0.0
        return obs

    def reset(self):
        start_position = self.start_position
        self.occupancy = self.generate_maze(self.seed)
        self.agt_pos_list = []
        for i in range(self.N_agent):
            self.agt_pos_list.append(copy.deepcopy(start_position[i % len(start_position)]))
        
        # get obs_0
        globel_obs = self.get_global_obs()
        if self.partical_obs != 1:
            map = np.ones([self.map_size + self.partical_obs*2-2,self.map_size + self.partical_obs*2-2,3])
            map[self.partical_obs-1:-self.partical_obs+1,self.partical_obs-1:-self.partical_obs+1,:] = globel_obs
        else:
            map = globel_obs
        
        map = (map[:,:,0]*0.5+map[:,:,1]*0.3+map[:,:,2]*0.2).squeeze()
        obs = []
        for i in range(self.N_agent):
                per_obs = np.concatenate((np.array([i/self.N_agent,self.agt_pos_list[i][0]/self.map_size,self.agt_pos_list[i][1]/self.map_size]),map[self.agt_pos_list[i][0]-1:self.agt_pos_list[i][0]+2*self.partical_obs,self.agt_pos_list[i][1]-1:self.agt_pos_list[i][1]+2*self.partical_obs].ravel()))
                obs.append(per_obs[np.newaxis,:])
       
        return obs

    def render(self):
        obs = self.get_global_obs()
        enlarge = 5
        new_obs = np.ones((self.map_size*enlarge, self.map_size*enlarge, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i * enlarge + enlarge, j * enlarge + enlarge), (0, 0, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i * enlarge + enlarge, j * enlarge + enlarge), (0, 0, 255), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i * enlarge + enlarge, j * enlarge + enlarge), (0, 255, 0), -1)
        cv2.imshow('image', new_obs)
        cv2.waitKey(10)


class EnvCleaner_oneimage(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,env_config):
        
        self.map_size = env_config["map_size"]
        self.seed = env_config["seed"]
        self.occupancy = self.generate_maze(self.seed)
        self.N_agent = env_config["N_agent"]
        self.agt_pos_list = []
        self.partical_obs = env_config["partical_obs"]
        for i in range(self.N_agent):
            self.agt_pos_list.append([1, 1])
        self.np_random = random.Random()
        self.action_space = spaces.Tuple([spaces.Discrete(5) for _ in range(self.N_agent)])
        # self.state_space = spaces.Box(low=-1, high=1, shape=(state_size,), dtype="float32")
        # obs shape = id+pos+(self.partical_obs*2+1)*(self.partical_obs*2+1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.N_agent,(self.partical_obs*2+1),(self.partical_obs*2+1)), dtype=np.float64) 
        self.start_position = env_config["start_position"]
        self._action_meanings = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT"]
        self.single_action_space = 5
        self.max_step = 200
        
        
    def get_action_meanings(self):
        return self._action_meanings
        
          
    def generate_maze(self, seed):
        symbols = {
            # default symbols
            'start': 'S',
            'end': 'X',
            'wall_v': '|',
            'wall_h': '-',
            'wall_c': '+',
            'head': '#',
            'tail': 'o',
            'empty': ' '
        }
        #生成地图的关键，改变地图形式可以参考第一句
        maze_obj = maze.Maze(int((self.map_size - 1) / 2), int((self.map_size - 1) / 2), seed, symbols, 1)
        grid_map = maze_obj.to_np()
        
        for i in range(self.map_size):
            for j in range(self.map_size):
                if grid_map[i][j] == 0:
                    grid_map[i][j] = 2
        return grid_map

      
    def step(self, action_list):
        done = False
        #Calculate individual reward
        self.in_reward = []
        self.info["game_step"] += 1
        if self.info["game_step"] >= self.max_step:
            done = True
        total_reward = 0
        if isinstance(action_list, (int, np.int64)):
            action_list = [action_list] 
        for i in range(len(action_list)):
            if action_list[i] == 1:     # up
                if self.occupancy[self.agt_pos_list[i][0] - 1][self.agt_pos_list[i][1]] != 1:  # if can move
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] - 1
            if action_list[i] == 2:     # down
                if self.occupancy[self.agt_pos_list[i][0] + 1][self.agt_pos_list[i][1]] != 1:  # if can move
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] + 1
            if action_list[i] == 3:     # left
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] - 1] != 1:  # if can move
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] - 1
            if action_list[i] == 4:     # right
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] + 1] != 1:  # if can move
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] + 1
            if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] == 2:   # if the spot is dirty
                self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] = 0
                self.in_reward.append(1.)
                total_reward += 1        
            else:
                self.in_reward.append(0.)
        total_reward -= 0.1
        self.info["global_reward"] += total_reward
        # Calculate pensonal observation with three channel
        globel_obs = self.get_global_obs()
        if self.partical_obs != 1:
            map = 0.01*np.ones([self.map_size + self.partical_obs*2-2,self.map_size + self.partical_obs*2-2,3])
            map[self.partical_obs-1:-self.partical_obs+1,self.partical_obs-1:-self.partical_obs+1,:] = globel_obs
        else:
            map = globel_obs
        map = (map[:,:,0]*0.5+map[:,:,1]*0.3+map[:,:,2]*0.2).squeeze()
        obs = []
        self.info["agent_info"] =[]
        for i in range(len(action_list)):
                per_obs = map[self.agt_pos_list[i][0]-1:self.agt_pos_list[i][0]+2*self.partical_obs,self.agt_pos_list[i][1]-1:self.agt_pos_list[i][1]+2*self.partical_obs]
                obs.append(per_obs)
                self.info["agent_info"].append([i/self.N_agent,self.agt_pos_list[i][0]/self.map_size,self.agt_pos_list[i][1]/self.map_size])
        # Calculate done
        if 2 not in self.occupancy:
            done = True


        return obs, total_reward, done, self.info

    def get_global_obs(self):
        obs = np.zeros((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.occupancy[i, j] == 0:#clean path
                    obs[i, j, 0] = 0.0
                    obs[i, j, 1] = 0.0
                    obs[i, j, 2] = 1.0
                if self.occupancy[i, j] == 1:#wall
                    obs[i, j, 0] = 0.0
                    obs[i, j, 1] = 0.0
                    obs[i, j, 2] = 0.05    
                if self.occupancy[i, j] == 2:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 0.0
                    obs[i, j, 2] = 0.0
        for i in range(self.N_agent):
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 0] = 1.0
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 1] = 1.0
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 2] = 1.0
        return obs

    def reset(self):
        start_position = self.start_position
        self.occupancy = self.generate_maze(self.seed)
        self.agt_pos_list = []
        self.info = {"global_reward":0,"game_step": 0,"agent_info":torch.zeros([self.N_agent,3])}
        for i in range(self.N_agent):
            self.agt_pos_list.append(copy.deepcopy(start_position[i % len(start_position)]))
            self.occupancy[self.start_position[i][0],self.start_position[i][1]]=0
            self.info["agent_info"][i] = torch.tensor([i,self.start_position[i%len(self.start_position)][0],self.start_position[i%len(self.start_position)][1]])
            
        # get obs_0
        globel_obs = self.get_global_obs()
        if self.partical_obs != 1:
            map = np.ones([self.map_size + self.partical_obs*2-2,self.map_size + self.partical_obs*2-2,3])
            map[self.partical_obs-1:-self.partical_obs+1,self.partical_obs-1:-self.partical_obs+1,:] = globel_obs
        else:
            map = globel_obs
        
        map = (map[:,:,0]*0.5+map[:,:,1]*0.3+map[:,:,2]*0.2).squeeze()
        obs = []
        for i in range(self.N_agent):
                per_obs = map[self.agt_pos_list[i][0]-1:self.agt_pos_list[i][0]+2*self.partical_obs,self.agt_pos_list[i][1]-1:self.agt_pos_list[i][1]+2*self.partical_obs]
                obs.append(per_obs)
        obs = np.stack(obs, axis=0 )
        return obs

    def render(self):
        obs = self.get_global_obs()
        enlarge = 50
        new_obs = np.ones((self.map_size*enlarge, self.map_size*enlarge, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i * enlarge + enlarge, j * enlarge + enlarge), (0, 0, 0), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.05:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i * enlarge + enlarge, j * enlarge + enlarge), (0, 0, 255), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i * enlarge + enlarge, j * enlarge + enlarge), (0, 255, 0), -1)
        cv2.imshow('image', new_obs)
        cv2.waitKey(2)

class EnvCleaner(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,env_config):
        
        self.map_size = env_config["map_size"]
        self.seed = env_config["seed"]
        self.occupancy = self.generate_maze(self.seed)
        self.N_agent = env_config["N_agent"]
        self.agt_pos_list = []
        self.partical_obs = env_config["partical_obs"]
        for i in range(self.N_agent):
            self.agt_pos_list.append([1, 1])
        self.np_random = env_config["seed"]
        self.action_space = spaces.Tuple([spaces.Discrete(4) for _ in range(self.N_agent)])
        # self.state_space = spaces.Box(low=-1, high=1, shape=(state_size,), dtype="float32")
        self.observation_space = spaces.Tuple([spaces.Box(low=0, high=1, shape=(self.partical_obs*2+1,self.partical_obs*2+1, 3), dtype=np.float64) for _ in range(self.N_agent)])
        self.start_position = [[1,1]]
        
        
        
    def generate_maze(self, seed):
        symbols = {
            # default symbols
            'start': 'S',
            'end': 'X',
            'wall_v': '|',
            'wall_h': '-',
            'wall_c': '+',
            'head': '#',
            'tail': 'o',
            'empty': ' '
        }
        #生成地图的关键，改变地图形式可以参考第一句
        maze_obj = maze.Maze(int((self.map_size - 1) / 2), int((self.map_size - 1) / 2), seed, symbols, 1)
        grid_map = maze_obj.to_np()
        
        for i in range(self.map_size):
            for j in range(self.map_size):
                if grid_map[i][j] == 0:
                    grid_map[i][j] = 2
        return grid_map

    def _step(self, action_list):
        reward = 0
        for i in range(len(action_list)):
            if action_list[i] == 0:     # up
                if self.occupancy[self.agt_pos_list[i][0] - 1][self.agt_pos_list[i][1]] != 1:  # if can move
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] - 1
            if action_list[i] == 1:     # down
                if self.occupancy[self.agt_pos_list[i][0] + 1][self.agt_pos_list[i][1]] != 1:  # if can move
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] + 1
            if action_list[i] == 2:     # left
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] - 1] != 1:  # if can move
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] - 1
            if action_list[i] == 3:     # right
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] + 1] != 1:  # if can move
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] + 1
            if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] == 2:   # if the spot is dirty
                self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] = 0
                reward = reward + 1
        return reward
    
    def step(self, action_list):
        done = False
        #Calculate individual reward
        self.in_reward = []
        total_reward = 0
        for i in range(len(action_list)):
            if action_list[i] == 0:     # up
                if self.occupancy[self.agt_pos_list[i][0] - 1][self.agt_pos_list[i][1]] != 1:  # if can move
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] - 1
            if action_list[i] == 1:     # down
                if self.occupancy[self.agt_pos_list[i][0] + 1][self.agt_pos_list[i][1]] != 1:  # if can move
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] + 1
            if action_list[i] == 2:     # left
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] - 1] != 1:  # if can move
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] - 1
            if action_list[i] == 3:     # right
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] + 1] != 1:  # if can move
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] + 1
            if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] == 2:   # if the spot is dirty
                self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] = 0
                self.in_reward.append(1.)
                total_reward += 1
            else:
                self.in_reward.append(0.)
        # Calculate pensonal observation with three channel
        globel_obs = self.get_global_obs()
        if self.partical_obs != 1:
            map = np.ones([self.map_size + self.partical_obs*2-2,self.map_size + self.partical_obs*2-2,3])
            map[self.partical_obs-1:-self.partical_obs+1,self.partical_obs-1:-self.partical_obs+1,:] = globel_obs
        else:
            map = globel_obs
        obs = []
        for i in range(len(action_list)):
                obs.append(map[self.agt_pos_list[i][0]-1:self.agt_pos_list[i][0]+2*self.partical_obs,self.agt_pos_list[i][1]-1:self.agt_pos_list[i][1]+2*self.partical_obs,:])
        
        # Calculate pensonal observation with one channel
        # if self.partical_obs != 1:
        #     map = np.ones([self.map_size + self.partical_obs*2-2,self.map_size + self.partical_obs*2-2])
        #     map[self.partical_obs-1:-self.partical_obs+1,self.partical_obs-1:-self.partical_obs+1] = self.occupancy
        # else:
        #     map = self.occupancy
        # obs = []
        # for i in range(len(action_list)):
        #     obs.append(map[self.agt_pos_list[i][0]-1:self.agt_pos_list[i][0]+2*self.partical_obs,self.agt_pos_list[i][1]-1:self.agt_pos_list[i][1]+2*self.partical_obs])
        
        # Calculate done
        if 2 not in self.occupancy:
            done = True
        info = {}
        
        
        return obs, total_reward, done, info

    def get_global_obs(self):
        obs = np.zeros((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.occupancy[i, j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                if self.occupancy[i, j] == 2:
                    obs[i, j, 0] = 0.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 0.0
        for i in range(self.N_agent):
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 0] = 1.0
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 1] = 0.0
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 2] = 0.0
        return obs

    def reset(self):
        start_position = self.start_position
        self.occupancy = self.generate_maze(self.seed)
        self.agt_pos_list = []
        for i in range(self.N_agent):
            self.agt_pos_list.append(copy.deepcopy(start_position[i % len(start_position)]))
        
        # get obs_0
        globel_obs = self.get_global_obs()
        if self.partical_obs != 1:
            map = np.ones([self.map_size + self.partical_obs*2-2,self.map_size + self.partical_obs*2-2,3])
            map[self.partical_obs-1:-self.partical_obs+1,self.partical_obs-1:-self.partical_obs+1,:] = globel_obs
        else:
            map = globel_obs
        obs = []
        for i in range(self.N_agent):
                obs.append(map[self.agt_pos_list[i][0]-1:self.agt_pos_list[i][0]+2*self.partical_obs,self.agt_pos_list[i][1]-1:self.agt_pos_list[i][1]+2*self.partical_obs,:])
        return obs

    def render(self):
        obs = self.get_global_obs()
        enlarge = 5
        new_obs = np.ones((self.map_size*enlarge, self.map_size*enlarge, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i * enlarge + enlarge, j * enlarge + enlarge), (0, 0, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i * enlarge + enlarge, j * enlarge + enlarge), (0, 0, 255), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (i * enlarge, j * enlarge), (i * enlarge + enlarge, j * enlarge + enlarge), (0, 255, 0), -1)
        cv2.imshow('image', new_obs)
        cv2.waitKey(10)

