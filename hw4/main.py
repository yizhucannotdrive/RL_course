import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def sample_jester_data(file_name, context_dim = 32, num_actions = 8, num_contexts = 19181,shuffle_rows=True, shuffle_cols=False):
    np.random.seed(0)
    with tf.gfile.Open(file_name, 'rb') as f:
        dataset = np.load(f)
    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)
        dataset = dataset[:num_contexts, :]
    assert context_dim + num_actions == dataset.shape[1]
    opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards = np.array([dataset[i, context_dim + a] for i, a in enumerate(opt_actions)])
    return dataset, opt_rewards, opt_actions

if __name__ == '__main__':
    filename = "datafile.npy"
    dataset, opt_rewards, opt_actions = sample_jester_data(filename)
    d = 32
    alpha = 0.3
    num_a = 8
    A = np.identity(d)
    b= np.zeros(d)
    A_inv = np.linalg.inv(A)
    theta = A_inv.dot(b)
    thetas = [theta] * num_a
    As = [A] * num_a
    bs = [b] * num_a
    A_invs = [A_inv] * num_a
    T_train = 19181
    rewards = []
    for i in range(T_train):
        feature =  dataset[i,:d]
        pta = []
        for a in range(num_a):
            pta.append(thetas[a].dot(feature) + alpha * np.sqrt(feature.dot(A_invs[a].dot(feature))))
        action = np.argmax(pta)
        reward = dataset[i,d+action]
        feature_mat = np.matrix(feature)
        As[action] += np.dot(feature_mat.transpose(), feature_mat)
        A_invs[action] = np.linalg.inv(As[action])
        bs[action] += feature * reward
        thetas[action] = A_invs[action].dot(bs[action])
        rewards.append(reward)
        print((i,action,reward, opt_rewards[i], opt_rewards[i] - reward))
        if opt_rewards[i] - reward>10:
            print(dataset[i])
    plot_int = [18000, 19181]
    optreward_show = np.cumsum(opt_rewards[plot_int[0]:plot_int[1]])
    rewards_show = np.cumsum(np.array(rewards[plot_int[0]:plot_int[1]]))
    regret = optreward_show -  rewards_show
    ms = np.arange(plot_int[0], plot_int[1], 1)
    plt.plot(ms, regret, 'r--',)
    plt.xlabel('iteration')
    plt.ylabel('regret')
    plt.show()

