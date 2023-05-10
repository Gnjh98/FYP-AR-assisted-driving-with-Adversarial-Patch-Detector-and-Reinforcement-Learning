import numpy as np
import torch
import gym
from Policy_2 import PPO, device
#from Policy_v1 import PPO, device
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from yolorl_33_sep import Walk
#from env.cartpole import CartPoleEnv
#from env.walk_v1 import Walk
#from env.yolorl_34_sep import Walk
#from env.action_decode import action_map


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--write', type=bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=400, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T_horizon', type=int, default=10000, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=1000000, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=5e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.97, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=512, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=1e-5, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-5, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=0.1, help='L2 regulization coefficient for Critic')
parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.92, help='Decay rate of entropy_coef')
opt = parser.parse_args()
print(opt)

def Action_adapter(a, min_action, max_action):
    #from [0,1] to [10,max]
    return  a * (max_action-min_action)+min_action


def evaluate_policy(env, model, min_action, max_action, render): # max for continuous for clamp
    #scores_u, scores_d, scores_g = 0, 0, 0
    turns = 3
    stepsTaken = 0

    full_r_u_d = 0
    delays = 0
    accuracies = 0
    dissatisfactions = 0
    for j in range(turns):
        done, ep_r_u_d, steps = False, 0, 0,
        s_ue, s_data = env.reset()

        delay_eps = 0
        accuracies_eps = 0
        dissatisfactions_eps = 0
        #print('start')
        while not done:

            # Take deterministic actions at test time
            a_d, prob_d = model.evaluate(torch.from_numpy(s_ue).float().to(device), 'down')
            a_u, log_u = model.evaluate(s_data, 'up')
            a_u = Action_adapter(a_u, min_action, max_action)  # continuous action space needs shaping

            #s_d_next, s_u_next, done, _, r_u_d, delay, accuracy, dissatisfaction = env.down_step(a_d, a_u)
            s_d_next, s_u_next, done, _, r_u_d, r_u_u, delay, accuracy, dissatisfaction = env.down_step(a_d, a_u)

            #print(r_u)
            #print(r_d)
            #print(r_g)
            # MODIFY!!!!!!!!!!!!!!!!!!!!!!!!!
            ep_r_u_d += r_u_d
            steps += 1
            s_u = s_u_next
            s_d = s_d_next
            #print('episode ends')
            delay_eps += delay
            accuracies_eps += accuracy
            dissatisfactions_eps += dissatisfaction

        full_r_u_d += ep_r_u_d
        delays += delay_eps
        accuracies += accuracies_eps
        dissatisfactions += dissatisfactions_eps
        stepsTaken += steps
    return full_r_u_d /turns, stepsTaken/turns, delays/turns, accuracies/turns, dissatisfactions/turns

def main():

    write = opt.write   #Use SummaryWriter to record the training.
    render = opt.render

    #EnvName = ['BipedalWalker-v3','BipedalWalkerHardcore-v3','LunarLanderContinuous-v2','Pendulum-v1','Humanoid-v2','HalfCheetah-v2']
    #BriefEnvName = ['BWv3', 'BWHv3', 'Lch_Cv2', 'PV0', 'Humanv2', 'HCv2']
    #Env_With_Dead = [True, True, True, False, True, False]
    #EnvIdex = opt.EnvIdex
    env_with_Dead = True  #Env like 'LunarLanderContinuous-v2' is with Dead Signal. Important!
    env = Walk()
    eval_env = Walk()
    state_d_dim = env.observation_space_d.shape[0]
    action_d_dim = env.action_space_d.n
    state_u_dim = env.observation_space_u.shape[0]
    action_u_dim = env.action_space_u.shape[0]
    min_action_u = float(env.action_space_u.low[0])
    max_action_u = float(env.action_space_u.high[0])
    max_steps = env.max_step
    print('  state_u_dim:',state_u_dim,  '  action_u_dim:',action_u_dim,
          '  state_d_dim', state_d_dim,  '  action_d_dim',action_d_dim,
          '  max_a_u:',    max_action_u, '  min_a_u:', env.action_space_u.low[0],
          '  max_steps', max_steps)
    T_horizon = opt.T_horizon  #lenth of long trajectory

    Max_train_steps = opt.Max_train_steps
    save_interval = opt.save_interval  # in steps
    eval_interval = opt.eval_interval  # in steps

    random_seed = opt.seed
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    eval_env.seed(random_seed)
    np.random.seed(random_seed)

    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format("lr{}").format(opt.a_lr) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    else:
        print('write file was not created')

    kwargs = {
        "state_u_dim": state_u_dim,
        "state_d_dim": state_d_dim,
        "action_u_dim": action_u_dim,
        "action_d_dim": action_d_dim,
        "env_with_Dead": env_with_Dead,
        "gamma": opt.gamma,
        "lambd": opt.lambd,     # For GAE
        "clip_rate": opt.clip_rate,  # 0.2
        "K_epochs": opt.K_epochs,
        "net_width": opt.net_width,
        "a_lr": opt.a_lr,
        "c_lr": opt.c_lr,
        "l2_reg": opt.l2_reg,   # L2 regulization for Critic
        "a_optim_batch_size":opt.a_optim_batch_size,
        "c_optim_batch_size": opt.c_optim_batch_size,
        "entropy_coef":opt.entropy_coef,  # Entropy Loss for Actor: Large entropy_coef for large exploration, but is harm for convergence.
        "entropy_coef_decay":opt.entropy_coef_decay
    }

    if not os.path.exists('model'): os.mkdir('model')
    model = PPO(**kwargs)
    if opt.Loadmodel: model.load(opt.ModelIdex)

    traj_lenth = 0
    total_steps = 0
    while total_steps < Max_train_steps:

        done, ep_r_u_d, steps = False, 0, 0,
        s_d, s_u = env.reset()

        delay_eps = 0
        accuracies_eps = 0
        dissatisfactions_eps = 0
        #print('start')
        while not done:
            traj_lenth += 1
            steps += 1
            # Take deterministic actions at test time
            #print('s_d shape is {}'.format(s_d.shape))
            a_d, prob_d = model.evaluate(torch.from_numpy(s_d).float().to(device), 'down')

            #print('s_u shape is {}'.format(s_u.shape))
            a_u, log_u = model.evaluate(s_u, 'up')
            a_u = Action_adapter(a_u, min_action_u, max_action_u)  # continuous action space needs shaping

            #s_d_next, s_u_next, done, _, r_u_d, delay, accuracy, dissatisfaction = env.down_step(a_d, a_u)
            s_d_next, s_u_next, done, _, r_u_d, r_u_u, delay, accuracy, dissatisfaction = env.down_step(a_d, a_u)
            #print('s_u_next shape is {}'.format(s_u_next.shape))
            if done and steps != max_steps:
                dw = True
            else:
                dw = False

            # data: 13 classes
            model.put_data((s_d, a_d, r_u_d, s_d_next, prob_d, s_u, a_u, r_u_d, s_u_next, log_u,
                            done, dw))
            s_d = s_d_next
            s_u = s_u_next
            ep_r_u_d += r_u_d

            accuracies_eps += accuracy
            delay_eps += delay
            dissatisfactions_eps += dissatisfaction

            '''update if its time'''

            if traj_lenth % T_horizon == 0:
                policy_u_loss, policy_d_loss, value_u_loss, value_d_loss = model.train()
                traj_lenth = 0
                '''
                if write:
                    
                    writer.add_scalar('policy_u_loss', policy_u_loss, global_step=total_steps)
                    writer.add_scalar('policy_d_loss', policy_d_loss, global_step=total_steps)
                    writer.add_scalar('value_u_loss', value_u_loss, global_step=total_steps)
                    writer.add_scalar('value_d_loss', value_d_loss, global_step=total_steps)
                    writer.add_scalar('mean_A_precision', accuracies_eps/steps, global_step=total_steps)
                    writer.add_scalar('delay', delay_eps, global_step=total_steps)
                    writer.add_scalar('dissatisfactions', dissatisfactions_eps, global_step=total_steps)
                    # writer.add_scalar('value_g_loss', value_g_loss, global_step=total_steps)
                '''

            '''record & log'''
            if total_steps % eval_interval == 0:
                score = evaluate_policy(eval_env, model, min_action_u, max_action_u, render)
                if write:
                    writer.add_scalar('ep_r_u', score[0], global_step=total_steps)
                    writer.add_scalar('delay', score[2], global_step=total_steps)
                    writer.add_scalar('mAp', score[3], global_step=total_steps)
                    writer.add_scalar('dissatisfaction', score[4], global_step=total_steps)

                #print('seed:',random_seed,'steps: {}k'.format(int(total_steps/1000)), 'stepsTakenPerEpisode: {}'.format(score[2]), 'reward_up: {}, reward_down: {}'.format(score[0],score[1]), 'delay down: {}, earning ability: {}'.format(score[3],score[4]), 'delay_up: {}, battery % consumption: {}'.format(score[5],score[6]))
                print('seed:',random_seed,'steps: {}k'.format(int(total_steps/1000)), 'stepsTakenPerEpisode: {}'.format(score[1]), 'reward: {}'.format(score[0]))
            total_steps += 1

            '''save model'''
            if total_steps % save_interval == 0:
                model.save(total_steps)

    env.close()

if __name__ == '__main__':
    main()