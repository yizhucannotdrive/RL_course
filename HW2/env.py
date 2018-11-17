import gym
import gym.spaces
from multiprocessing import Process, Pipe
import numpy  as np

def make_envs(args):
    return EnvPool([make_env(args.game) for i in range(args.envs)])


def make_env(gamename):
    def _thunk():
        env = gym.make(gamename)
        return env
    return _thunk


def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'spec':
            remote.send((env.observation_spec(), env.action_spec()))
        elif cmd == 'step':
            #print(data)
            #exit()
            obs = env.step(data)
            remote.send(obs)
        elif cmd == 'reset':
            state = env.reset()
            remote.send(state)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'save_replay':
            env.save_replay(data)
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class EnvPool(object):
    def __init__(self, env_fns):
        nenvs = len(env_fns)
        #self.remotes: parent connections; self.work_remotes: child connections
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

    def spec(self):
        for remote in self.remotes:
            remote.send(('spec', None))
        results = [remote.recv() for remote in self.remotes]
        # todo maybe support running different envs / specs in the future?
        return results[0]

    def step(self, actions):
        #print(actions)
        for remote, action in zip(self.remotes, actions):
            #print(action)
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        return results

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
            #here is the problem
            #print("!!!!!!!!!!!!!!")
            #print(self.remotes[0].recv())
            #exit()
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def save_replay(self, replay_dir='gameReplays'):
        self.remotes[0].send(('save_replay', replay_dir))

    @property
    def num_envs(self):
        return len(self.remotes)
