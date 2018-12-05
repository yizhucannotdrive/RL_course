import time
import numpy as np
from baselines import logger
from common import flatten_lists
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
class Runner:
    def __init__(self, envs, agent, n_steps=8, game = 'Pong-v0', buffer_size = 1000000, batchsize = 32):
        self.state = self.logs = self.ep_rews = None
        self.agent, self.envs, self.n_steps = agent, envs, n_steps
        self.episode_print = 0
        self.game = game
        self.episode = 0
        self.buffer_size =  buffer_size
        self.batchsize = batchsize
    def run(self, num_updates=1, train=True):
        self.reset()
        self.frames = 0
        replay_buffer = dict()
        replay_buffer["states"] = deque(maxlen = self.buffer_size)
        replay_buffer["rewards"] = deque(maxlen = self.buffer_size)
        replay_buffer["actions"] = deque(maxlen = self.buffer_size)
        curr_epi_return = 0
        for i in range(num_updates):
            print(i)
            self.episode =i
            self.logs['updates'] += 1
            self.logs['ep_rew'] = np.zeros(self.envs.num_envs)
            self.state = self.envs.reset()
            if self.game == 'Breakout-v0':
                self.preprocess()
            if self.game == 'MsPacman-v0':
                self.preprocess()

            #if rollout[-1]==200:
            #    print (i)
            #    break
            #print(len(rollout[0]))
            #print(len(rollout[1]))
            #print(len(rollout[2]))
            #print(rollout[1])
            #print(rollout[2])
            epi_reward = 0
            real_epi_reward = 0
            while True:
                rollout = self.collect_rollout()
                epi_reward += sum(rollout[2])
                real_epi_reward += sum(rollout[4])
                replay_buffer["states"].extend(rollout[0])
                replay_buffer["actions"].extend(rollout[1])
                replay_buffer["rewards"].extend(rollout[2])
                #print("hello world")
                #exit()
                if train and i>50:
                    if len(replay_buffer["states"]) < self.batchsize:
                        self.agent.train(i, replay_buffer["states"], replay_buffer["actions"], replay_buffer["rewards"],  curr_epi_return)
                    else:
                        batch_index = np.random.permutation(np.arange(len(replay_buffer["states"])))[:self.batchsize]
                        self.agent.train(i, list(np.array(replay_buffer["states"])[batch_index]), list(np.array(replay_buffer["actions"])[batch_index]), list(np.array(replay_buffer["rewards"])[batch_index]), curr_epi_return)
                if sum(rollout[3])==1:
                    curr_epi_return = 0.9 * curr_epi_return + 0.1 * epi_reward
                    print("episode reward: {}".format(epi_reward))
                    print("real_episode reward: {}".format(real_epi_reward))
                    break

    def preprocess(self):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
        if self.game =="Breakout_v0":
            for i  in range(len(self.state)):
                image = self.state[i]
                image = image[35:195]  # crop
                image = image[::2, ::2, 0]  # downsample by factor of 2
                image[image == 144] = 0  # erase background (background type 1)
                image[image == 109] = 0  # erase background (background type 2)
                image[image != 0] = 1  # everything else just set to 1
                self.state[i] = image.astype(np.float).ravel()
        if self.game =="MsPacman-v0":
            mspacman_color = 210 + 164 + 74
            for i in range(len(self.state)):
                img = self.state[i][1:176:2, ::2]  # crop and downsize
                img = img.sum(axis=2)  # to greyscale
                img[img == mspacman_color] = 0  # Improve contrast
                img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
                self.state[i]=img.reshape(88, 80, 1)



    def collect_rollout(self):
        #states, actions = [None]*self.n_steps, [None]*self.n_steps
        #rewards, dones, values = np.zeros((3, self.n_steps, self.envs.num_envs))
        states = []
        actions = []
        rewards =[]
        dones = []
        pre_states = None
        self.episode_print = 0
        self.frames =0
        real_rewards = []
        for step in range(self.n_steps):
            if self.game=='Breakout-v0':
                statediff = np.array(self.state) - pre_states if pre_states is not None else np.zeros(np.array(self.state).shape)
                pre_states  = np.array(self.state)
                action = self.agent.act(statediff)
                states.append(statediff)
                actions.append(action)
                #print(np.array(action)+2)
                #actionhere=  [np.random.randint(0,4)]
                actionhere = action
                self.state, rew, do = self.envs.step(actionhere)
                #if rew[0]!=0:
                #    print (actionhere)
                #    print("rew",rew)
                self.preprocess()
            elif self.game =='MsPacman-v0':
                action = self.agent.act(self.state)
                states.append(self.state)
                actions.append(action)
                self.state, rew, do = self.envs.step(action)
                self.preprocess()
                # print("rew",rew)
            else:
                action = self.agent.act(self.state)
                states.append(self.state)
                actions.append(action)
                self.state, rew, do = self.envs.step(action)

                #print("rew",rew)
            real_rewards.append(rew)
            if sum(do) >= 1:
                rew = [0]
            else:
                rew = [1]

            rewards.append(rew)
            dones.append(do)
            #self.log(step, rewards[-1], dones[-1])
            if sum(dones[step]) >= 1:
                break
        states = [j for i in states for j in i]
        rewards = [j for i in rewards for j in i]
        actions =[ j for i in actions for j in i]
        real_rewards = [j  for i in real_rewards for j in i]
        #print(rewards)
        #print(actions)
        if self.game == "Breakout-v0":
            return states, actions, rewards, np.array(dones), real_rewards
        else:
            return states, actions, real_rewards, np.array(dones), real_rewards


    def reset(self):
        self.state = self.envs.reset()
        #print(self.state)
        self.logs = {'updates': 0, 'eps': 0, 'rew_best': -21, 'start_time': time.time(),
                     'ep_rew': np.zeros(self.envs.num_envs), 'dones': np.zeros(self.envs.num_envs)}
    """
    def log(self, step, rewards, dones):
        if self.episode_print != 1:
            self.logs['ep_rew'] += np.asarray(rewards)
        #if np.asarray(rewards)[0]!=0:
        #    print(self.logs['ep_rew'])
        #print(self.logs['ep_rew'] )
        checkdone = np.copy(self.logs['dones'])
        #if step ==0:
        #    print(self.logs['dones'])
        self.logs['dones'] = np.maximum(self.logs['dones'], dones)
        #if step == self.n_steps -1:
        #    print(step==self.n_steps-1 and self.episode_print == 0)
        #if (sum(self.logs['dones']) == self.envs.num_envs and self.episode_print == 0) or  (step==self.n_steps-1 and self.episode_print == 0) :
        if (sum(self.logs['dones']) >= 1 and self.episode_print == 0) or (
                    step == self.n_steps - 1 and self.episode_print == 0):
            if sum(self.logs['dones']) != self.envs.num_envs:
                print("not done")
            self.episode_print = 1
            #print(step)
            # print (sum(self.logs['dones']))
            # print(self.logs['ep_rew'] )
            # exit()
            self.logs['eps'] += self.envs.num_envs
            self.logs['rew_best'] = max(self.logs['rew_best'], np.mean(self.logs['ep_rew']))

            elapsed_time = time.time() - self.logs['start_time']
            # print(self.envs.num_envs, self.n_steps, self.logs['updates'])
            #print(self.ep_rews, np.mean(self.logs['ep_rew']), self.episode)
            self.ep_rews = np.mean(self.logs['ep_rew']) if self.ep_rews is None else 0.9 * self.ep_rews + 0.1*np.mean(self.logs['ep_rew'])
            logger.logkv('fps', int(self.frames / elapsed_time))
            logger.logkv('elapsed_time', int(elapsed_time))
            logger.logkv('n_eps', self.logs['eps'])
            logger.logkv('n_samples', self.frames)
            logger.logkv('n_updates', self.logs['updates'])
            logger.logkv('rew_best_mean', self.logs['rew_best'])
            # logger.logkv('rew_max', np.max(self.logs['ep_rew']))
            if np.mean(self.logs['ep_rew'])==0 or np.mean(self.logs['ep_rew']) < -21:
                print(self.logs['dones'])
                print(self.logs['ep_rew'])
                print(step)
                print(rewards)
                print(dones)
                print(checkdone)
                exit()
            logger.logkv('rew_mean_current', self.logs['ep_rew'] )
            logger.logkv('rew_mean', self.ep_rews )
            # logger.logkv('rew_mestd', np.std(self.logs['ep_rew'])) # weird name to ensure it's above min since logger sorts
            # logger.logkv('rew_min', np.min(self.logs['ep_rew']))
            logger.dumpkvs()
            return
        else:
            return
    """