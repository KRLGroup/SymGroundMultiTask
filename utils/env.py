"""
This class defines the environments that we are going to use.
Note that this is the place to include the right LTL-Wrapper for each environment.
"""


import gym
# import gym_minigrid
# import envs.gym_letters
import envs
from envs.gridworld_multitask.Environment import LTLWrapper as GridWorldLTLWrapper
import ltl_wrappers

def make_env(env_key, progression_mode, ltl_sampler, seed=None, intrinsic=0, noLTL=False):
    env = gym.make(env_key)
    env.seed(seed)

    # Adding LTL wrappers
    if (noLTL):
        return ltl_wrappers.NoLTLWrapper(env)
    if "GridWorld" in env_key:
        assert ltl_sampler == 'None'
        return GridWorldLTLWrapper(env, progression_mode, ltl_sampler, intrinsic)
    return ltl_wrappers.LTLEnv(env, progression_mode, ltl_sampler, intrinsic)
