import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled

"""
一个bs多个user, uplink和downlink数据量不同，Uplink和downlink干扰不同(但相似)
downlink超过时限重传
"""
def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


# user type
User = np.dtype({
    'names': ['num', 'p_d', 'g', 'channel', 'dissatisfaction', 'data', 'x', 'y', 'rate'],
    'formats': ['int', 'float', 'float', 'int', 'float', 'float', 'float', 'float', 'float']
})
BaseStation = np.dtype({
    'names': ['x', 'y'],
    'formats': ['float','float']
})

def action_map(x, usr, chal):
    num = np.zeros(usr)
    res = np.zeros(usr + 1)
    res[0] = x
    for i in range(usr):
        num[usr-i-1] = res[i]//(chal**(usr-1-i))
        res[i+1] = res[i] % (chal**(usr-1-i))
    return num.astype(np.int32)


class Walk(gym.Env):

    def __init__(self):
        # 6 users, 3 base
        self.max_step = 100
        self.count = 0
        self.comp = 0
        self.down_state = None
        self.user_num = 3
        self.channel_num = 3
        # band width
        self.B = 5e6
        self.sigma = 9.9e-7
        self.wavelength = 1
        self.tau = 0.5
        self.user_loc = np.random.uniform(0, 10, size=(self.user_num + self.channel_num, 2))
        # ['x', 'y']
        self.channel_loc = np.array([(self.user_loc[3][0], self.user_loc[3][1]),
                                     (self.user_loc[4][0], self.user_loc[4][1]),
                                     (self.user_loc[5][0], self.user_loc[5][1])
                                     ], dtype=BaseStation)
        # self.channel_loc = np.random.uniform(0, 10, size=2)
        # user [num, p, p_d, g, data, channel, rate, x, y, comp]
        # self.data = np.random.randint(6e7, 1e8, size=self.user_num)
        self.data = np.random.randint(32, 320, size=self.user_num)
        #self.batteryLife = np.random.randint(35000000, 40000000, size=self.user_num)
        #self.batteryLife = np.random.randint(35000000, 40000000, size=self.user_num)
        # ['num', 'p_u', 'g', 'channel', 'dissatisfaction', data', 'x', 'y']
        self.user = np.array([(1, 3.0, 2.0, 0, 0, self.data[0], self.user_loc[0][0], self.user_loc[0][1], 0),
                              (2, 4.0, 3.8, 0, 0, self.data[1], self.user_loc[1][0], self.user_loc[1][1], 0),
                              (3, 3.5, 1.4, 0, 0, self.data[2], self.user_loc[2][0], self.user_loc[2][1], 0)
                              ]
                             , dtype=User)

        self.channel = list(range(1, self.channel_num + 1))

        self.down_select = 0

        self.down_action_num = (self.channel_num+1) ** self.user_num
        self.action_space_d = spaces.Discrete(self.down_action_num)
        self.observation_space_d = spaces.Box(
            low=np.array([0 for _ in range(5 * self.user_num)], dtype=np.float32),
            high=np.array([np.finfo(np.float32).max for _ in range(5 * self.user_num)], dtype=np.float32)
        )

        # uplink allocate power to base station
        self.action_space_u = spaces.Box(
            low=48, high=320, shape=(self.user_num,), dtype=np.float32
        )
        self.observation_space_u = spaces.Box(
            low=np.array([0 for _ in range(5 * self.user_num)], dtype=np.float32),
            high=np.array([np.finfo(np.float32).max for _ in range(5 * self.user_num)], dtype=np.float32)
        )

        self.down_state = None
        self.step_beyond_done = None
        self.up_state = None
        self.group = dict()
        self.steps_beyond_done = None
        self.done = False
        self.battery_die = False

    def down_step(self, ue_action, data_action):
        c = action_map(int(ue_action), self.user_num, self.channel_num+1)  # select [c1, c2, c3...]

        # data transferred, channel
        for i in range(self.user_num):
            self.user[i]['data'] = data_action[i]

        # distance from user to channel
        for i in range(self.user_num):

            if c[i] != 0:
                dist = np.linalg.norm(np.array(self.user[i]['x'], self.user[i]['y'])
                                      - np.array(self.channel_loc[c[i]-1][0], self.channel_loc[c[i]-1][1]))
                # debatable equation - check again
                self.user[i]['g'] = (self.wavelength / (4 * math.pi * dist)) ** 2
            else:
                self.user[i]['g'] = 0


        # self.data_trans = [0 for _ in range(self.user_num)]
        self.down_select = int(ue_action)
        # give base information
        for i in range(self.user_num):
            self.user[i]['channel'] = c[i]

        # divide users by channel, self.group里没有0
        for channel in self.channel:
            self.group[channel] = []
            for u in self.user:
                if u['channel'] == channel:
                    self.group[channel].append(u)

        interfere = [self.B * self.sigma ** 2 for _ in range(self.user_num)]


        # calculate downlink time
        for i in range(self.user_num):
            # intra-cell interference
            if self.user[i]['channel'] != 0:
                for u_other in self.group[self.user[i]['channel']]:
                    if u_other['num'] != self.user[i]['num']:
                        #interfere[i] += u_other['p_u'] * u_other['g']
                        interfere[i] += u_other['p_d'] * self.user[i]['g']
                #print('downlink power to user is {}'.format(self.user[i]['p_d']))
                #print('channel gain is {}'.format(self.user[i]['g']))
                self.user[i]['rate'] = self.B * np.log2(
                    1 + self.user[i]['p_d'] * self.user[i]['g'] / interfere[i])
                #print('data transfer rate is {}'.format(self.user[i]['rate']))

        '''we didn't consider inter-cell (interference by other cells to user i)'''
        '''Downlink latency'''
        utility_ds = 0
        utility_us = 0
        delays = 0
        accuracies = 0
        dissatisfaction = 0
        # latency_d = 0
        alpha = 50 # This is the weighting constant for the objective function
        P = 5 # This is constant for play-2-earn
        k = 50e7 # This is player ability factor
        weighting1 = 0.50 # This is to adjust weight on earning ability and delay
        weighting2 = 1.0
        '''
        desirable params
        alpha = 50, multiply = 10, disatisfaction 50, clamp -5 5
        alpha = 50, multiply = 10, disatisfaction 5, clamp -5 5
        alpha = 70, multiply = 10, disatisfaction 5, clamp -5 5
        '''
        for i in range(self.user_num):
            utility_d = 0
            utility_u = 0
            if self.user[i]['channel'] != 0:
                self.user[i]['dissatisfaction'] = 0
                delay = self.user[i]['data']**2 / self.user[i]['rate']
                a, b = -1.23646287, 0.35627271
                accuracy = a + b*np.log(self.user[i]['data'])

                if accuracy != 0 or accuracy != None:
                    utility_u = 10*(weighting1 * 2 * alpha * delay - (1-weighting1) * 2 *accuracy) / self.user_num
                else:
                    utility_u = 10*(weighting1 *2* (alpha * delay) / self.user_num)

                utility_d = 10*(weighting2 *2* (alpha * delay) / self.user_num)
                delays += delay/self.user_num
                accuracies += accuracy/self.user_num

                #print('delay is {}'.format(delay))
                #print('accuracy is {}'.format(accuracy))
                #print('resolution is {}'.format(self.user[i]['data']))
            else: # channel == 0
                utility_d = 0
                self.user[i]['rate'] = 0
                self.user[i]['dissatisfaction'] += 1
                if self.user[i]['dissatisfaction'] == 1:
                    utility_d += ((1-weighting2) * 2 *5)/self.user_num
                    dissatisfaction -= ((1-weighting1) * 2 *5)/self.user_num
                    #print('dissatisfaction is {}'.format(dissatisfaction))
            #print('utility_d is {}'.format(utility_d))
            utility_ds -= clamp(utility_d, -5/self.user_num, 5/self.user_num)
            utility_us -= clamp(utility_u, -5/self.user_num, 5/self.user_num)
            #print('utility_ds is {}'.format(utility_ds))

        self.ue_state = []
        self.data_state = []
        for i in range(self.user_num):
            self.ue_state.append(self.user[i]['data'])
            self.ue_state.append(self.user[i]['dissatisfaction'])
            self.ue_state.append(self.user[i]['x'])
            self.ue_state.append(self.user[i]['y'])
            self.ue_state.append(self.user[i]['g'])

            self.data_state.append(self.user[i]['data'])
            self.data_state.append(self.user[i]['dissatisfaction'])
            self.data_state.append(self.user[i]['x'])
            self.data_state.append(self.user[i]['y'])
            self.data_state.append(self.user[i]['g'])

        for i in range(self.user_num):
            self.user[i]['x'] += np.random.uniform(-1, 1)
            self.user[i]['y'] += np.random.uniform(-1, 1)
            # new x and y of users
            self.user[i]['x'] = self.clamp(self.user[i]['x'], 0, 10)
            self.user[i]['y'] = self.clamp(self.user[i]['y'], 0, 10)

        # every time step
        self.count += 1
        if self.count >= self.max_step:
            self.done = True
        #print(utility_ds)
        return np.array(self.ue_state,dtype=np.float32), np.array(self.data_state, dtype=np.float32), self.done, {}, utility_ds, utility_us, delays/self.max_step, accuracies/self.max_step, dissatisfaction/self.max_step

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):


        self.user = np.array([(1, 3.0, 2.0, 0, 0, self.data[0], self.user_loc[0][0], self.user_loc[0][1], 0),
                              (2, 4.0, 3.8, 0, 0, self.data[1], self.user_loc[1][0], self.user_loc[1][1], 0),
                              (3, 3.5, 1.4, 0, 0, self.data[2], self.user_loc[2][0], self.user_loc[2][1], 0)
                              ]
                             , dtype=User)

        self.ue_state = []
        self.data_state = []

        for i in range(self.user_num):

            self.ue_state.append(self.user[i]['data'])
            self.ue_state.append(self.user[i]['dissatisfaction'])
            self.ue_state.append(self.user[i]['x'])
            self.ue_state.append(self.user[i]['y'])
            self.ue_state.append(self.user[i]['g'])

            self.data_state.append(self.user[i]['data'])
            self.data_state.append(self.user[i]['dissatisfaction'])
            self.data_state.append(self.user[i]['x'])
            self.data_state.append(self.user[i]['y'])
            self.data_state.append(self.user[i]['g'])

        self.count = 0
        self.steps_beyond_done = None
        self.group = dict()
        self.done = False

        return np.array(self.ue_state, dtype=np.float32), np.array(self.data_state, dtype=np.float32)

    def render(self, mode="human"):
        pass

    # clamping users motion to within the designated area
    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)