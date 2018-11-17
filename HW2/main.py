import os, argparse
from absl import flags
import tensorflow as tf

from common import Config
from common.env import make_envs
from rl.agent import A2CAgent
from rl.model import fully_conv
from rl.model import carpole_net
from rl import Runner, EnvWrapper


if __name__ == '__main__':
    flags.FLAGS(['main.py'])
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--updates", type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--vf_coef', type=float, default=0.15)
    parser.add_argument('--ent_coef', type=float, default=5e-4)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--clip_grads', type=float, default=1.)
    parser.add_argument("--run_id", type=int, default=-1)
    parser.add_argument("--game", type=str, default='Pong-v0')
    parser.add_argument("--test", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--restore", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--agenttype", type=str, default="vpg") # or you can use "a2c"
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tf.reset_default_graph()
    sess = tf.Session()

    envs = EnvWrapper(make_envs(args))
    # not specify which feature to use when initialization
    if args.game == 'Pong-v0':
        num_action = 3
        input_size =  80
        agent = A2CAgent(sess, fully_conv, input_size, num_action, args.game,  args.restore, args.discount, args.lr, args.vf_coef, args.ent_coef, args.clip_grads, agenttype = agenttype)
        runner = Runner(envs, agent, args.steps, args.game, agenttype = agenttype)
        runner.run(args.updates, not args.test)
    else:
        num_action = 2
        input_size = None
        agent = A2CAgent(sess, carpole_net, input_size, num_action, args.game, args.restore, args.discount, args.lr,
                         args.vf_coef, args.ent_coef, args.clip_grads)
        runner = Runner(envs, agent, args.steps, args.game)
        runner.run(args.updates, not args.test)

    envs.close()


