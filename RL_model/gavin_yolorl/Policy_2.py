import copy
import numpy as np
import torch
from torch.distributions import Categorical
import math
from model import DiscretePolicy, ContinousPolicy, HybridValue
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO(object):
    def __init__(self,
                 state_u_dim,
                 state_d_dim,
                 action_u_dim,
                 action_d_dim,
                 env_with_Dead,
                 gamma=0.99,
                 lambd=0.95,
                 clip_rate=0.2,
                 K_epochs=10,
                 net_width=256,
                 a_lr=3e-4,
                 c_lr=3e-4,
                 l2_reg=1e-3,
                 a_optim_batch_size=64,
                 c_optim_batch_size=64,
                 entropy_coef=0,
                 entropy_coef_decay=0.9998
                 ):
        # self.actor_up = DiscretePolicy(state_u_dim, action_u_dim, net_width).to(device)
        self.actor_up = ContinousPolicy(state_u_dim, action_u_dim, net_width).to(device)
        # self.actor_down = ContinousPolicy(state_d_dim, action_d_dim, net_width).to(device)
        self.actor_down = DiscretePolicy(state_d_dim, action_d_dim, net_width).to(device)
        self.actor_up_optimizer = torch.optim.Adam(self.actor_up.parameters(), lr=a_lr)
        self.actor_down_optimizer = torch.optim.Adam(self.actor_down.parameters(), lr=a_lr)

        self.critic = HybridValue(state_u_dim, state_d_dim, net_width).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        self.env_with_Dead = env_with_Dead
        self.action_u_dim = action_u_dim
        self.action_d_dim = action_d_dim
        self.clip_rate = clip_rate
        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.data = []
        self.l2_reg = l2_reg
        self.a_optim_batch_size = a_optim_batch_size
        self.c_optim_batch_size = c_optim_batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay

    def select_action(self, state, stage):
        with torch.no_grad():
            if stage == 'down':
                pi = self.actor_down.pi(state, softmax_dim=0)
                m = Categorical(pi)
                a = m.sample().item()
                pi_a = pi[a].item()
                return a, pi_a
            elif stage == 'up':
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                dist = self.actor_up.get_dist(state)
                a = dist.sample()
                a = torch.clamp(a, 0, 1)
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
                return a.cpu().numpy().flatten(), logprob_a
            else:
                raise NotImplementedError('Unknown stage {}'.format(stage))

    def evaluate(self, state, stage):
        with torch.no_grad():
            if stage == 'down':
                pi = self.actor_down.pi(state, softmax_dim=0)
                a = torch.argmax(pi).item()
                return a, 1.0
            elif stage == 'up':
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                #state = state.reshape((1, -1))
                #state = state.to(device)
                dist = self.actor_up.get_dist(state)
                a = dist.sample()
                a = torch.clamp(a, 0, 1)
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
                return a.cpu().numpy().flatten(), logprob_a
            else:
                raise NotImplementedError('Unknown stage {}'.format(stage))

    def train(self):
        self.entropy_coef *= self.entropy_coef_decay
        s_d, a_d, r_d, s_d_next, old_prob_d, s_u, a_u, r_u, s_u_next, old_log_prob_u, terminal, dw = self.make_batch()

        # Use TD+GAE+LongTrac to compute Advantage and TD target
        with torch.no_grad():
            # uplink critic
            vu = self.critic(s_u, "up")
            vu_ = self.critic(s_u_next, "up")
            adv_u, td_target_u = self.get_adv_td(vu, vu_, r_u, terminal, dw)

            # downlink critic
            vd = self.critic(s_d, "down")
            vd_ = self.critic(s_d_next, "down")
            adv_d, td_target_d = self.get_adv_td(vd, vd_, r_d, terminal, dw)

            # global critic
            '''
            vg = self.critic((s_u, s_d), "global")
            vg_ = self.critic((s_u_next, s_d_next), "global")
            adv_g, td_target_g = self.get_adv_td(vg, vg_, r_g, terminal, dw)
            '''
        # slice long traj into short traj and perform mini-batch PPO update
        # NLP+RL里两个actor也都是用的user state shape[0]
        a_optim_iter_num = int(math.ceil(s_u.shape[0] / self.a_optim_batch_size))
        c_optim_iter_num = int(math.ceil(s_d.shape[0] / self.c_optim_batch_size))
        for i in range(self.K_epochs):
            # shuffle the traj, Good for training
            perm = np.arange(s_u.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            s_d, a_d, td_target_d, adv_d, prob_d, s_u, a_u, td_target_u, adv_u, log_prob_u = \
                s_d[perm].clone(), a_d[perm].clone(), td_target_d[perm].clone(), adv_d[perm].clone(), old_prob_d[perm].clone(), \
                s_u[perm].clone(), a_u[perm].clone(), td_target_u[perm].clone(), adv_u[perm].clone(), old_log_prob_u[perm].clone(), \
                #td_target_g[perm].clone(), adv_g[perm].clone()

            policy_u_loss, policy_d_loss, value_u_loss, value_d_loss = 0., 0., 0., 0.

            # update the actor
            for i in range(a_optim_iter_num):
                index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s_u.shape[0]))
                # downlink actor
                #print('index is {}'.format(index))
                #print('s_d[index] is {}'.format(s_d[index]))
                prob_d = self.actor_down.pi(s_d[index], softmax_dim=1)
                #print('prob_d is {}'.format(prob_d))
                entropy_d = Categorical(prob_d).entropy().sum(0, keepdim=True)
                #print('entropy_d is {}'.format(entropy_d))

                prob_a_d = prob_d.gather(1, a_d[index].long())
                ratio_d = torch.exp(torch.log(prob_a_d) - torch.log(old_prob_d))
                # !!!
                surr1_d = ratio_d * (adv_d[index])
                surr2_d = torch.clamp(ratio_d, 1 - self.clip_rate, 1 + self.clip_rate) * (adv_d[index])

                actor_d_loss = -torch.min(surr1_d, surr2_d) - self.entropy_coef * entropy_d

                policy_d_loss += actor_d_loss.mean()
                self.actor_down_optimizer.zero_grad()
                actor_d_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor_down.parameters(), 40)
                self.actor_down_optimizer.step()

                '''This is where we combine the loss of both uplink and downlink model if we want to add their losses together'''

                #uplink actor
                distribution = self.actor_up.get_dist(s_u[index])
                dist_entopy = distribution.entropy().sum(1, keepdim=True)
                logprob_now = distribution.log_prob(a_u[index])
                ratio_u = torch.exp(logprob_now.sum(1, keepdim=True) - old_log_prob_u[index].sum(1, keepdim=True))
                # !!!
                surr1_u = ratio_u * (adv_u[index])
                surr2_u = torch.clamp(ratio_u, 1 - self.clip_rate, 1 + self.clip_rate) * (adv_u[index])

                actor_u_loss = -torch.min(surr1_u, surr2_u) - self.entropy_coef * dist_entopy
                policy_u_loss += actor_u_loss.mean()
                #('policy_u_loss is {}'.format(policy_u_loss))
                self.actor_up_optimizer.zero_grad()
                actor_u_loss.mean().backward()
                #print('actor_u_loss is {}'.format(actor_u_loss))
                torch.nn.utils.clip_grad_norm_(self.actor_up.parameters(), 40)
                self.actor_up_optimizer.step()


            # update the critic
            for i in range(c_optim_iter_num):
                index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s_u.shape[0]))

                critic_u_loss = (self.critic(s_u[index], 'up') - td_target_u[index]).pow(2).mean()
                critic_d_loss = (self.critic(s_d[index], 'down') - td_target_d[index]).pow(2).mean()
                #critic_g_loss = (self.critic((s_u[index], s_d[index]), 'global') - td_target_g[index]).pow(2).mean()
                critic_loss = critic_u_loss + critic_d_loss # + critic_g_loss
                '''
                we already have the critic to take into account both losses here
                '''
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        critic_loss += param.pow(2).sum() * self.l2_reg
                value_u_loss += critic_u_loss.item()
                value_d_loss += critic_d_loss.item()
                # value_g_loss += critic_g_loss.item()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            policy_u_loss /= a_optim_iter_num # policy loss
            policy_d_loss /= a_optim_iter_num
            value_u_loss /= c_optim_iter_num #critic loss
            value_d_loss /= c_optim_iter_num
            #value_g_loss /= c_optim_iter_num
            return policy_u_loss, policy_d_loss, value_u_loss, value_d_loss#, value_g_loss

    def get_adv_td(self, vs, vs_, r, terminal, dw):
        '''dw for TD_target and Adv'''
        deltas = r + self.gamma * vs_ * (1 - dw) - vs

        deltas = deltas.cpu().flatten().numpy()
        adv = [0]

        '''done for GAE'''
        for dlt, mask in zip(deltas[::-1], terminal.cpu().flatten().numpy()[::-1]):
            advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
            adv.append(advantage)
        adv.reverse()
        adv = copy.deepcopy(adv[0:-1])
        adv = torch.tensor(adv).unsqueeze(1).float().to(device)
        td_target = adv + vs
        #adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  # sometimes helps
        return adv, td_target

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        # state: up, down, up_next, down_next
        # action: up, down
        # reward: up, down, global
        # logprob_a: up, down
        # terminal
        # dead or win mask
        # 13 in total
        s_d_lst, a_d_lst, r_d_lst, s_d_next_lst, prob_d_lst, s_u_lst, a_u_lst, r_u_lst, s_u_next_lst, log_prob_u_lst, terminal_lst, dw_lst =\
        [], [], [], [], [], [], [], [], [], [], [], []
        for transition in self.data:
            s_d, a_d, r_d, s_d_next, prob_d, s_u, a_u, r_u, s_u_next, log_prob_u, terminal, dw = transition
            s_d_lst.append(s_d)
            a_d_lst.append([a_d]) # aware: [a] not a!!!!
            r_d_lst.append([r_d])
            s_d_next_lst.append(s_d_next)
            prob_d_lst.append(prob_d)
            s_u_lst.append(s_u)
            a_u_lst.append([a_u])
            r_u_lst.append([r_u])
            s_u_next_lst.append(s_u_next)
            log_prob_u_lst.append(log_prob_u)
            # r_g_lst.append([r_g])
            terminal_lst.append([terminal])
            dw_lst.append([dw])

        if not self.env_with_Dead:
            # env_without_DeadAndWin: deltas = r + self.gamma * vs_ - vs
            # env_with_DeadAndWin: deltas = r + self.gamma * vs_ * (1 - dw) - vs
            dw_lst = (np.array(dw_lst) * False).tolist()

        self.data = []

        with torch.no_grad():
            s_d, a_d, r_d, s_d_next, prob_d, s_u, a_u, r_u, s_u_next, log_prob_u, terminal, dw = \
                torch.tensor(s_d_lst, dtype=torch.float).to(device), \
                torch.tensor(a_d_lst, dtype=torch.float).to(device), \
                torch.tensor(r_d_lst, dtype=torch.float).to(device), \
                torch.tensor(s_d_next_lst, dtype=torch.float).to(device), \
                torch.tensor(prob_d_lst, dtype=torch.float).to(device), \
                torch.tensor(s_u_lst, dtype=torch.float).to(device), \
                torch.tensor(a_u_lst, dtype=torch.float).to(device), \
                torch.tensor(r_u_lst, dtype=torch.float).to(device), \
                torch.tensor(s_u_next_lst, dtype=torch.float).to(device), \
                torch.tensor(log_prob_u_lst, dtype=torch.float).to(device), \
                torch.tensor(terminal_lst, dtype=torch.float).to(device), \
                torch.tensor(dw_lst, dtype=torch.float).to(device),

        # here are all lists
        return s_d, a_d, r_d, s_d_next, prob_d, s_u, a_u, r_u, s_u_next, log_prob_u, terminal, dw

    def save(self, episode):
        torch.save(self.actor_up.state_dict(), "./model/actor_up{}.pth".format(episode))
        torch.save(self.actor_down.state_dict(), "./model/actor_down{}.pth".format(episode))
        torch.save(self.critic.state_dict(), "./model/critic{}.pth".format(episode))

    def load(self, episode):
        self.critic.load_state_dict(torch.load("./model/critic{}.pth".format(episode)))
        self.actor_up.load_state_dict(torch.load("./model/actor_up{}.pth".format(episode)))
        self.actor_down.load_state_dict(torch.load("./model/actor_down{}.pth".format(episode)))