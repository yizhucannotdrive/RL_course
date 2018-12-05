import os, argparse
from absl import flags
import tensorflow as tf

from common.env import make_envs
from rl.DQNagent import DQNAgent
from rl.model import carpole_net_local
from rl.model import carpole_net_target
from rl.model import Breakout_local
from rl.model import Breakout_target
from rl.model import MsPacman_local
from rl.model import MsPacman_target
from rl import Runner, EnvWrapper


if __name__ == '__main__':
    flags.FLAGS(['main.py'])
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--updates", type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--buffer_size', type=int, default=20000)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--sync_duration', type=int, default=10)
    parser.add_argument('--epsilon_initial', type=float, default=1.0)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--clip_grads', type=float, default=1.)
    parser.add_argument("--game", type=str, default='CartPole-v0')
    parser.add_argument("--test", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--restore", type=bool, nargs='?', const=True, default=False)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tf.reset_default_graph()
    sess = tf.Session()

    envs = EnvWrapper(make_envs(args))
        # not specify which feature to use when initialization
    if  args.game == 'CartPole-v0':
        num_action = 2
        input_size = None
        agentmodel_local = carpole_net_local
        agentmodel_target = carpole_net_target
    if  args.game == 'Breakout-v0':
        num_action = 4
        input_size = 80 * 80
        agentmodel_local = Breakout_local
        agentmodel_target = Breakout_target
    if  args.game == 'MsPacman-v0':
        num_action = 9
        input_size = [88,80]
        agentmodel_local = MsPacman_local
        agentmodel_target = MsPacman_target
    agent = DQNAgent(sess, agentmodel_local, agentmodel_target , input_size, num_action, args.game, args.restore, args.discount, args.lr, args.clip_grads,  args.epsilon_initial, args.epsilon_decay, sync_duration = args.sync_duration)
    runner = Runner(envs, agent, args.steps, args.game, buffer_size =  args.buffer_size, batchsize =  args.batchsize)
    runner.run(args.updates, not args.test)
    envs.close()



