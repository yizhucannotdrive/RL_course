
class EnvWrapper:
    def __init__(self, envs):
        self.envs = envs

    def step(self, acts):
        acts = self.wrap_actions(acts)
        results = self.envs.step(acts)
        return self.wrap_results(results)

    def reset(self):
        results = self.envs.reset()
        return results
    #this is used by step
    def wrap_actions(self, actions):
        return actions

    def wrap_results(self, results):
        obs = [res[0] for res in results]
        rewards = [res[1] for res in results]
        dones = [res[2] for res in results]
        states = obs

        return states, rewards, dones

    def save_replay(self, replay_dir='gameReplays'):
        self.envs.save_replay(replay_dir)

    def spec(self):
        return self.envs.spec()

    def close(self):
        return self.envs.close()

    @property
    def num_envs(self):
        return self.envs.num_envs