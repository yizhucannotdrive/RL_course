import numpy as np
import pickle as pickle
import gym
import matplotlib.pyplot as plt

num_units= 200
batch_size = 10
learning_rate = 1e-4
Dim = 80 * 80
gamma = 0.99
decay_rate = 0.99


def sigmoid_func(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def preprocess(Image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    Image = Image[35:195]  # crop
    Image = Image[::2, ::2, 0]  # downsample by factor of 2
    Image[Image == 144] = 0  # erase background (background type 1)
    Image[Image == 109] = 0  # erase background (background type 2)
    Image[Image != 0] = 1  # everything else (paddles, ball) just set to 1
    return Image.astype(np.float).ravel()


def normalize(discounted_epr):
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    return discounted_epr

def act_get_prob(x, model):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid_func(logp)
    return p, h  # return probability of taking action 2, and hidden state


def derive_derivative(eph, epdlogp, model):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}



if __name__ == '__main__':
    total_episodes = 20000
    model = {}
    model['W1'] = np.random.randn(num_units, Dim) / np.sqrt(Dim)
    model['W2'] = np.random.randn(num_units) / np.sqrt(num_units)
    grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}
    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0
    batch_baseine = None
    running_rewards = []
    while episode_number < total_episodes:
        current_x = preprocess(observation)
        x = current_x - prev_x if prev_x is not None else np.zeros(Dim)
        prev_x = current_x
        aprob, h = act_get_prob(x, model)
        action = 2 if np.random.uniform() < aprob else 3  # roll the dice!
        xs.append(x)
        hs.append(h)
        fake_label = 1 if action == 2 else 0
        dlogps.append(
            fake_label - aprob)
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward)
        if done:
            episode_number += 1
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], []
            discounted_epr = discount_rewards(epr)
            discounted_epr = normalize(discounted_epr)
            epdlogp *= discounted_epr
            grad = derive_derivative(eph, epdlogp, model)
            for k in model:
                grad_buffer[k] += grad[k]

            if episode_number % batch_size == 0:
                for k, v in model.iteritems():
                    g = grad_buffer[k]
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v)
            running_reward = reward_sum if running_reward is None else running_reward * 0.98 + reward_sum * 0.02
            print ('episode%f : episode reward total %f. running average: %f' % (episode_number, reward_sum, running_reward))
            if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
            reward_sum = 0
            observation = env.reset()
            prev_x = None
            running_rewards.append(running_reward)
    episodes = range(total_episodes)
    plt.plot(episodes, running_rewards, 'ro')
    plt.xlabel('episodes')
    plt.ylabel('running_reward')
    plt.show()
