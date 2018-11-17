import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import os


class A2CAgent:
    def __init__(self, sess, model_fn, input_size, num_action, game, restore=False, discount=0.99, lr=1e-4, vf_coef=0.25, ent_coef=1e-3, clip_grads=1., agenttype = "vpg"):
        self.sess, self.discount = sess, discount
        self.vf_coef, self.ent_coef = vf_coef, ent_coef
        self.game = game
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.agenttype = agenttype

        if game == "Pong-v0":
            (self.policy, self.value), self.inputs = model_fn(input_size, num_action)
            #print(sample(self.policy))
            self.action = sample(self.policy)
        else:
            (self.policy, self.value), self.inputs = model_fn(num_action)
            self.action = sample(self.policy)
        loss_fn, loss_val, self.loss_inputs = self._loss_func()

        self.step = tf.Variable(0, trainable=False)
        #opt = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5)
        opt = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
        #self.train_op = layers.optimize_loss(loss=loss_fn, optimizer=opt, learning_rate=None, global_step= self.global_step_tensor, clip_gradients=clip_grads)
        #self.train_op_val = layers.optimize_loss(loss=loss_val, optimizer=opt, learning_rate=None, global_step= self.global_step_tensor, clip_gradients=clip_grads)
        self.train_op = layers.optimize_loss(loss=loss_fn, optimizer=opt, learning_rate=None,
                                             global_step=self.global_step_tensor)
        self.train_op_val = layers.optimize_loss(loss=loss_val, optimizer=opt, learning_rate=None,
                                                 global_step=self.global_step_tensor)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        if restore:
            self.saver.restore(self.sess, tf.train.latest_checkpoint('weights/' + self.game))

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('logs/' + self.game, graph=None)
        self.summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), sess.run(self.step))

    # TODO: get rid of the step param; gracefully restore for console logs as well
    def train(self, step, states, actions, returns, ep_rews, iftrain=False):
        if step % 500 == 0:
            directory = 'weights/%s/a2c' % self.game
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.saver.save(self.sess, directory, global_step=self.step)
        #returns =  np.array(rewards).flatten()
        #print(returns)
        #print(type(states), type(actions), type(returns))
        #print(actions,returns)
        #print(actions)
        #print(actions)
        #print(type(states))
        #print(actions, returns)
        #exit()
        #print(len(states),len(states[0]),len(states[0][0]))
        feed_dict = dict(zip(self.loss_inputs, [actions] + [returns]))
        feed_dict[self.inputs] = states
        #print(feed_dict[self.inputs])
        batchsize = len(actions)
        if iftrain:
            batches = len(actions)/batchsize +1 if len(actions)%batchsize != 0 else len(actions)/batchsize
            for i in range(int(batches)):
                states_batch = states[i*batchsize:(i+1)*batchsize]
                actions_batch = actions[i*batchsize:(i+1)*batchsize]
                returns_batch = returns[i*batchsize:(i+1)*batchsize]
                feed_dict = dict(zip(self.loss_inputs, [actions_batch] + [returns_batch]))
                feed_dict[self.inputs] = states_batch
                self.baselinevalue = self.sess.run(self.value, feed_dict={self.inputs: states_batch})
                result= self.sess.run([self.train_op], feed_dict)
                result_val= self.sess.run([self.train_op_val], feed_dict)
            print("trained")
            self.step+=1
        else:
            self.step+=1
        result_summary, step = self.sess.run([self.summary_op, self.step], feed_dict)
        print(step)
        #print(ep_rews)
        self.summary_writer.add_summary(summarize(rewards=ep_rews), global_step=step)
        self.summary_writer.add_summary(result_summary, step)

    def act(self, state):
        act, val, pol = self.sess.run([self.action, self.value, self. policy], feed_dict={self.inputs:state})
        #print(pol)
        return act, val

    def get_value(self, state):
        return self.sess.run(self.value, feed_dict={self.inputs:state})

    def _loss_func(self):
        returns = tf.placeholder(tf.float32, [None])
        actions = tf.placeholder(tf.int32, [None])
        #returnmean, returnvar = tf.nn.moments(returns, axes =0)
        #try this
        #adv = tf.stop_gradient(tf.div(returns-returnmean, tf.sqrt(returnvar)))
        if self.agenttype == "vpg":
            adv = tf.stop_gradient(returns)
        else:
            adv = tf.stop_gradient(returns - self.baselinevalue)
        pi = select(actions, self.policy)
        logli = clip_log(pi)
        #logli = pi * clip_log(pi)
        entropy = -tf.reduce_sum(self.policy * clip_log(self.policy), axis=-1)

        #policy_loss = -tf.reduce_mean(logli * returns)
        policy_loss = -tf.reduce_mean(tf.multiply(logli, adv))
        entropy_loss = -self.ent_coef * tf.reduce_mean(entropy)
        value_loss = self.vf_coef * tf.reduce_mean(tf.square(returns - self.value))
        tf.summary.scalar('loss/total', policy_loss + entropy_loss + value_loss)
        tf.summary.scalar('loss/policy', policy_loss)
        tf.summary.scalar('loss/entropy', entropy_loss)
        tf.summary.scalar('loss/value', value_loss)
        #print(policy_loss, entropy_loss, value_loss, actions, returns)
        #return [actions] + [returns]
        return policy_loss,value_loss,  [actions] + [returns]




def select(acts, policy):
    return tf.gather_nd(policy, tf.stack([tf.range(tf.shape(policy)[0]), acts], axis=1))


# based on https://github.com/pekaalto/sc2aibot/blob/master/common/util.py#L5-L11
def sample(probs):
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111")
    #print(probs.shape)
    #exit()
    #u = tf.random_uniform(tf.shape(probs))
    #ans = tf.argmax(tf.log(u) / probs, axis=1)
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111")
    #print(type(ans))
    #exit()
    ans = tf.squeeze(tf.multinomial(probs, 1), axis=[1])
    return ans


def clip_log(probs):
    return tf.log(tf.clip_by_value(probs, 1e-12, 1.0))


def summarize(**kwargs):
    summary = tf.Summary()
    for k, v in kwargs.items():
        summary.value.add(tag=k, simple_value=v)
    return summary
