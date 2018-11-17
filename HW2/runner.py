import time
import numpy as np
from baselines import logger
from common import flatten_lists
from PIL import Image
import matplotlib.pyplot as plt
class Runner:
    def __init__(self, envs, agent, n_steps=8, game = 'Pong-v0', agenttype = "vpg"):
        self.state = self.logs = self.ep_rews = None
        self.agent, self.envs, self.n_steps = agent, envs, n_steps
        self.episode_print = 0
        self.game = game
        self.batch  = 1
        self.episode = 0
        self.agenttype = agenttype

    def run(self, num_updates=1, train=True):
        # based on https://github.com/deepmind/pysc2/blob/master/pysc2/env/run_loop.py
        self.reset()
        self.frames = 0
        if self.game =='Pong-v0':
            self.preprocess()
        batch_obs = []
        batch_ret = []
        batch_act = []
        try:
            #print(num_updates)
            for i in range(num_updates):
                print(i)
                self.episode =i
                #if i==1:
                #    exit()
                #print(self.logs['dones'])
                #self.state = self.envs.reset()
                #print(self.logs['dones'])
                #self.preprocess()
                self.logs['updates'] += 1
                rollout = self.collect_rollout()
                self.logs['dones'] = np.zeros(self.envs.num_envs)
                self.logs['ep_rew'] = np.zeros(self.envs.num_envs)
                self.state = self.envs.reset()
                self.preprocess()
                #if rollout[-1]==200:
                #    print (i)
                #    break
                #print(len(rollout[0]))
                #print(len(rollout[1]))
                #print(len(rollout[2]))
                #print(rollout[1])
                #print(rollout[2])
                batch_obs+= rollout[0]
                batch_act+= rollout[1]
                batch_ret+= rollout[2]


                #print(rollout[-1])
                #exit()
                if train :
                    if i%self.batch==0:
                        #print(len(batch_ret))
                        if self.agenttype == 'vpg':
                            batch_ret_mean = np.mean(batch_ret)
                            batch_ret_std = np.std(batch_ret)
                            #batch_ret = (np.array(batch_ret)- batch_ret_mean)/batch_ret_std
                            batch_ret = np.array(batch_ret)- batch_ret_mean
                        #print(batch_ret_mean,batch_ret_std)
                        self.agent.train(i, batch_obs, batch_act, batch_ret, rollout[-1], iftrain = True)
                        batch_obs = []
                        batch_ret = []
                        batch_act = []
                    else:
                        self.agent.train(i, batch_obs, batch_act, batch_ret, rollout[-1], iftrain = False)

        except KeyboardInterrupt:
            pass
        finally:
            elapsed_time = time.time() - self.logs['start_time']
            frames = self.envs.num_envs * self.n_steps * self.logs['updates']
            print("Took %.3f seconds for %s steps: %.3f fps" % (elapsed_time, frames, frames / elapsed_time))

    def preprocess(self):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
        if self.game == 'Pong-v0':
            for i in range(len(self.state)):
                image = self.state[i][35:195] # crop
                image = image[::2,::2,0] # downsample by factor of 2
                image[image == 144] = 0 # erase background (background type 1)
                img = Image.fromarray(image)
                #img.show()
                image[image == 109] = 0 # erase background (background type 2)
                img = Image.fromarray(image)
                #img.show()
                image[image != 0] = 1 # everything else (paddles, ball) just set to 1
                img = Image.fromarray(image)
                #img.show()
                #self.state[i] = np.reshape(image.astype(np.float).ravel(), [80,80])
                img = Image.fromarray(self.state[i])
                #img.show()
                self.state[i] = image.astype(np.float).ravel()

    def collect_rollout(self):
        #states, actions = [None]*self.n_steps, [None]*self.n_steps
        #rewards, dones, values = np.zeros((3, self.n_steps, self.envs.num_envs))
        states = []
        actions = []
        rewards =[]
        dones = []
        values =[]
        pre_states = None
        self.episode_print = 0
        self.frames =0
        for step in range(self.n_steps):
            #if self.episode_print==1:
            #    print(step)
            #    exit()
            #if step == 1000:
            #    exit()
            #print(len(self.state), len(self.state[0]),  len(self.state[0]))
            #action, values[step] = self.agent.act(self.state)
            #print(action)
            #print(action, values[step])
            #exit()
            #states[step], actions[step] = self.state, list(action)
            #print(actions[step])
            #self.state, reward_step, dones[step] = self.envs.step(action)
            #rewards[step] = np.asarray(reward_step)
            #print(dones[step], rewards[step])
            #self.preprocess()
            if self.game=='Pong-v0':
                statediff = np.array(self.state) - pre_states if pre_states is not None else np.zeros(np.array(self.state).shape)
                pre_states  = np.array(self.state)
                #action, val = self.agent.act(self.state)
                action, val = self.agent.act(statediff)
                if action[0]!=0:
                    action[0]+=1
                #print(action)
                #states.append(self.state)
                states.append(statediff)
                values.append(val)
                actions.append(action)
                #print(action)
                self.state, rew, do = self.envs.step(action)
                #if step == 110:
                #    img = Image.fromarray(self.state[0], 'RGB')
                    #img.show()
                """
                               image = self.state[0][35:195]  # crop
                               print(image.shape)
                               image = image[::2,::2, 0]  # downsample by factor of 2
                               image[image == 144] = 0  # erase background (background type 1)
                               image[image == 109] = 0  # erase background (background type 2)
                               image[image != 0] = 1  # everything else (paddles, ball) just set to 1
                               #plt.imshow(image)
                               self.state[0] = np.reshape(image.astype(np.float).ravel(), [80, 80])
                               img = Image.fromarray(self.state[0])
                               #img.show()
                               statediff = np.array(self.state) - pre_states if pre_states is not None else np.zeros(
                                   np.array(self.state).shape)
                               plt.imshow(self.state[0] )
                               plt.show()
                               plt.imshow(pre_states[0])
                               plt.show()
                               plt.imshow(statediff[0])
                               plt.show()
                               exit()
                               for i in range(len(self.state[0])):
                                   if sum(self.state[0][i] != 0):
                                       print (self.state[0][i])
                               print("******************************************")
                               for i in range(len(statediff[0])):
                                   if sum(statediff[0][i] != 0):
                                       print (statediff[0][i])
                               exit()
                               """
                self.preprocess()
                last_value = self.agent.get_value(statediff)
            else:
                action, val = self.agent.act(self.state)
                states.append(self.state)
                values.append(val)
                actions.append(action)
                self.state, rew, do = self.envs.step(action)
                last_value = self.agent.get_value(self.state)

            rewards.append(rew)
            dones.append(do)
            self.frames += 1
            self.log(step, rewards[step], dones[step])
            #if sum(dones[step]) == self.envs.num_envs:
            if sum(dones[step]) >= 1:
                #print(step)
                break
        states = [j for i in states for j in i]
        #print(actions)
        #print(rewards)
        #print(len(states), len(states[0]), len(states[0][0]))
        #print(len(actions), len(actions[0]))

        #states = list(map(list,zip(*states)))
        #print(type(states))
        #print(len(states), len(states[0]), len(states[0][0]))
        #print(len(flatten_lists(actions)), len(flatten_lists(actions)[0]))
        #print(rewards)
        #exit()
        #print(len(flatten_lists(states)), len(flatten_lists(states)[0]), len(flatten_lists(states)[0][0]), len(flatten_lists(states)[0][0]))
        #exit()
        #print(self.logs['ep_rew'])
        test_state = states[100:101]
        #plt.imshow(statediff[0])
        #plt.show()
        #exit()
        returns = self._compute_returns( np.array(rewards), np.array(dones),last_value)
        actions =[ j for i in actions for j in i]
        #print(actions[:10])
        #print(returns)
        #exit()
        return states, actions, list(returns), np.array(dones), np.array(last_value), self.ep_rews

    def _compute_returns(self, rewards, dones, last_value):
        # print(rewards, dones, last_value)
        returns = np.zeros((dones.shape[0] + 1, dones.shape[1]))
        #returns[-1] = last_value.flatten()
        returns[-1] = np.zeros(dones.shape[1])
        for t in reversed(range(dones.shape[0])):
            """
            for i in len(rewards[t]):
                if rewards[t][i] == 0:
                    returns[t][i] = rewards[t][i] + self.discount * returns[t+1][i] * (1-dones[t])
                else:
                    returns[t][i] = rewards[t][i]
            """
            if self.game == 'Pong-v0':
                if rewards[t] == 0:
                    returns[t] = rewards[t] + 0.99 * returns[t + 1] * (1 - dones[t])
                else:
                    returns[t] = rewards[t]
            else:
                returns[t] = rewards[t] + 0.99 * returns[t + 1] * (1 - dones[t])
            # if rewards[t]==-1:
            #    print(returns[t], returns[t+1])
        returns = returns[:-1]
        # print(returns)
        # print(returns.flatten())
        # exit()
        # print(dones.shape[0])
        # print(len(returns))
        # print(returns[:10])
        # print(rewards[:1000])
        # exit()
        return returns.flatten()

    def reset(self):
        self.state = self.envs.reset()
        #print(self.state)
        self.logs = {'updates': 0, 'eps': 0, 'rew_best': -21, 'start_time': time.time(),
                     'ep_rew': np.zeros(self.envs.num_envs), 'dones': np.zeros(self.envs.num_envs)}

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
