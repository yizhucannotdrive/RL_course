import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import os


class DQNAgent:
    def __init__(self, sess, model_local, model_target, input_size, num_action, game, restore=False, discount=0.99, lr=1e-4, clip_grads=1., epslilon_initial = 1.0, epsilon_decay = 0.995, sync_duration = 1000):
        self.sess, self.discount = sess, discount
        self.epsilon = epslilon_initial
        self.epsilon_decay = epsilon_decay
        self.num_action = num_action
        self.game = game
        self.sync_duration = sync_duration
        if self.game == "CartPole-v0":
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.local_network, self.local_w0, self.local_wv, self.local_inputs = model_local(input_size, num_action)
            self.target_network, self.target_w0, self.target_wv, self.target_inputs = model_target(input_size, num_action)
            self.copyTargetQNetworkOperation = [self.target_w0.assign(self.local_w0),
                                                self.target_wv.assign(self.local_wv)]
        if self.game == "Breakout-v0":
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            #self.local_network, self.local_w0,  self.local_w1,  self.local_w2,  self.local_wv, self.local_inputs = model_local(input_size, num_action)
            #self.target_network, self.target_w0,  self.target_w1,  self.target_w2,  self.target_wv, self.target_inputs = model_target(input_size, num_action)
            self.local_network, self.local_w0,  self.local_wv, self.local_inputs = model_local(input_size, num_action)
            self.target_network, self.target_w0,  self.target_wv, self.target_inputs = model_target(input_size, num_action)
            self.copyTargetQNetworkOperation = [self.target_w0.assign(self.local_w0),
                                                self.target_wv.assign(self.local_wv),
                                                #self.target_w1.assign(self.local_w1),
                                                #self.target_w2.assign(self.local_w2)
                                                ]
        if self.game == "MsPacman-v0":
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.local_network, self.local_w0,  self.local_w1,  self.local_w2, self.local_w3,  self.local_wv, self.local_inputs = model_local(input_size, num_action)
            self.target_network, self.target_w0,  self.target_w1,  self.target_w2, self.target_w3,  self.target_wv, self.target_inputs = model_target(input_size, num_action)
            self.copyTargetQNetworkOperation = [self.target_w0.assign(self.local_w0),
                                                self.target_wv.assign(self.local_wv),
                                                self.target_w1.assign(self.local_w1),
                                                self.target_w2.assign(self.local_w2),
                                                self.target_w3.assign(self.local_w3)
                                                ]
        loss_val, self.loss_inputs = self._loss_func()
        self.step = tf.Variable(0, trainable=False)

        #opt = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5)
        opt = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
        self.train_op = layers.optimize_loss(loss=loss_val, optimizer=opt, learning_rate = None, global_step= self.global_step_tensor, clip_gradients=clip_grads)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        if restore:
            self.saver.restore(self.sess, tf.train.latest_checkpoint('weights/' + self.game))
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('logs/' + self.game, graph=None)
        self.summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), sess.run(self.step))

    # TODO: get rid of the step param; gracefully restore for console logs as well
    def train(self, step, states, actions, rewards, epi_return):
        if step % 10 == 0:
            self.epsilon = max([self.epsilon * self.epsilon_decay, 0.1])
            #self.epsilon = (1. - 0.05)/20000*step
            print(self.epsilon)
            directory = 'weights/%s/dqn' % self.game
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.saver.save(self.sess, directory, global_step=self.step)
        target_val = self.sess.run(self.target_network, feed_dict={self.target_inputs: states})
        ys = np.array(rewards) + self.discount* np.max(target_val, axis=1)
        feed_dict = dict(zip(self.loss_inputs, [actions] + [rewards] + [ys]))
        feed_dict[self.local_inputs] = states
        #pitest= self.sess.run([self.pi], feed_dict)
        #A3test= self.sess.run([self.A3], feed_dict)
        result= self.sess.run([self.train_op], feed_dict)
        self.step = self.step + 1
        result_summary, tenboard_step = self.sess.run([self.summary_op, self.step], feed_dict)
        if step % self.sync_duration == 0:
            target_wv, local_wv = self.sess.run([self.target_wv, self.local_wv], feed_dict={self.target_inputs: states})
            self.copyTargetQNetwork()
            target_wv_after, local_wv_after = self.sess.run([self.target_wv, self.local_wv], feed_dict={self.target_inputs: states})

        self.summary_writer.add_summary(summarize(Q_est=np.mean(ys)), global_step=tenboard_step)
        self.summary_writer.add_summary(summarize(rewards_current=epi_return), global_step=tenboard_step)
        self.summary_writer.add_summary(result_summary, tenboard_step)

    def copyTargetQNetwork(self):
        self.sess.run(self.copyTargetQNetworkOperation)
    def act(self,state):
        val = self.sess.run([self.local_network], feed_dict={self.local_inputs:state})
        if np.random.random() <= self.epsilon:
            action= np.random.randint(self.num_action)
        else:
            action = np.argmax(val)
        # change episilon
        #if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
        #    self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return [action]


    def _loss_func(self):
        returns = tf.placeholder(tf.float32, [None])
        actions = tf.placeholder(tf.int32, [None])
        ys = tf.placeholder(tf.float32, [None])
        q_vals = select(actions, self.local_network)
        value_loss = tf.reduce_mean(tf.square(ys - q_vals))
        tf.summary.scalar('loss/value', value_loss)
        return value_loss, [actions] + [returns]+[ys]


def select(acts, value):
    return tf.gather_nd(value, tf.stack([tf.range(tf.shape(value)[0]), acts], axis=1))


# based on https://github.com/pekaalto/sc2aibot/blob/master/common/util.py#L5-L11


def summarize(**kwargs):
    summary = tf.Summary()
    for k, v in kwargs.items():
        summary.value.add(tag=k, simple_value=v)
    return summary
