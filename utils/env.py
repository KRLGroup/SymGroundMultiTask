"""
This class defines the environments that we are going to use.
Note that this is the place to include the right LTL-Wrapper for each environment.
"""


import gym
import envs
from envs.gridworld_multitask.Environment import LTLWrapper as GridWorldLTLWrapper
from ltl_wrappers import NoLTLWrapper, LTLEnv



def make_env(env_key, progression_mode, ltl_sampler, seed=None, intrinsic=0, noLTL=False, grounder=None, obs_size=None):

    kwargs = {} if not "GridWorld" in env_key else {"grounder": grounder, "obs_size": obs_size}

    env = gym.make(env_key, **kwargs)
    env.seed(seed)

    if (noLTL):
        wrapper = NoLTLWrapper(env)

    elif "GridWorld" in env_key:
        wrapper = GridWorldLTLWrapper(
            env=env,
            progression_mode=progression_mode,
            ltl_sampler=ltl_sampler,
            intrinsic=intrinsic
        )

    else:
        wrapper = LTLEnv(
            env=env,
            progression_mode=progression_mode,
            ltl_sampler=ltl_sampler,
            intrinsic=intrinsic
        )

    return wrapper