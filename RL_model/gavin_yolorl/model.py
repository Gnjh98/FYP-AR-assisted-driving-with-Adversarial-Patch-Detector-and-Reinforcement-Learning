import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ContinousPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, log_std=0):
        super(ContinousPolicy, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        return mu

    def get_dist(self, state):
        mu = self.forward(state)
        action_log_std = self.action_log_std.expand_as(mu)
        action_std = torch.exp(action_log_std)

        dist = Normal(mu, action_std)
        return dist


class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(DiscretePolicy, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        #print(state.shape)
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim = 0):
        n = self.forward(state)
        #print('n is {}'.format(n))
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        #print('n is {}'.format(prob))
        return prob


class HybridValue(nn.Module):
    def __init__(self, up_state_dim, down_state_dim, net_width):
        super(HybridValue, self).__init__()

        self.Cu = nn.Linear(up_state_dim, net_width)
        self.Cd = nn.Linear(down_state_dim, net_width)
        self.Cg = nn.Linear(net_width+net_width, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, s, character):
        if character == 'up':
            v = torch.tanh(self.Cu(s))
            v = torch.tanh(self.C2(v))
            v = self.C3(v)
        elif character == 'down':
            v = torch.tanh(self.Cd(s))
            v = torch.tanh(self.C2(v))
            v = self.C3(v)
        elif character == 'global':
            h_s_up = self.Cu(s[0])
            h_s_down = self.Cd(s[1])
            h = torch.cat([h_s_up, h_s_down], -1)
            v = torch.tanh(self.Cg(h))
            v = torch.tanh(self.C2(v))
            v = self.C3(v)
        else:
            raise NotImplementedError('Unknown character {}'.format(character))
        return v