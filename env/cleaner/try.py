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
            map = 0.01*np.ones([self.map_size + self.partical_obs*2-2,self.map_size + self.partical_obs*2-2,3])
            map[self.partical_obs-1:-self.partical_obs+1,self.partical_obs-1:-self.partical_obs+1,:] = globel_obs
        else:
            map = globel_obs
        map = (map[:,:,0]*0.5+map[:,:,1]*0.3+map[:,:,2]*0.2).squeeze()
        obs = []
        for i in range(len(action_list)):
                per_obs = np.concatenate((np.array([i/self.N_agent,self.agt_pos_list[i][0]/self.map_size,self.agt_pos_list[i][1]/self.map_size]),map[self.agt_pos_list[i][0]-1:self.agt_pos_list[i][0]+2*self.partical_obs,self.agt_pos_list[i][1]-1:self.agt_pos_list[i][1]+2*self.partical_obs].ravel()))
                obs.append(per_obs[np.newaxis,:])
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