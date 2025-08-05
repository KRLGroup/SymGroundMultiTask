import numpy
import torch
import collections


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d


def average_reward_per_step(returns, num_frames):
    avgs = []
    assert(len(returns) == len(num_frames))

    for i in range(len(returns)):
        avgs.append(returns[i] / num_frames[i])

    return numpy.mean(avgs)


def average_discounted_return(returns, num_frames, disc):
    discounted_returns = []
    assert(len(returns) == len(num_frames))

    for i in range(len(returns)):
        discounted_returns.append(returns[i] * (disc ** (num_frames[i]-1)))

    return numpy.mean(discounted_returns)


def empty_episode_logs():
    logs = {
        "return_per_episode": [],
        "reshaped_return_per_episode": [],
        "num_frames_per_episode": [],
        "num_frames": 0
    }
    return logs


def empty_buffer_logs():
    logs = {
        'buffer': 0, 'val_buffer': 0, 
        'total_buffer': 0, 'total_val_buffer': 0
    }
    return logs


def empty_algo_logs():
    logs = {
        "entropy": 0.0,
        "value": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "grad_norm": 0.0
    }
    return logs


def empty_grounder_algo_logs():
    logs = {
        'grounder_loss': 0.0,
        "grounder_val_loss": 0.0
    }
    return logs


def accumulate_episode_logs(logs_exp, logs):

    logs_exp["return_per_episode"] += logs["return_per_episode"]
    logs_exp["reshaped_return_per_episode"] += logs["reshaped_return_per_episode"]
    logs_exp["num_frames_per_episode"] += logs["num_frames_per_episode"]
    logs_exp["num_frames"] = logs["num_frames"]

    return logs_exp


def elaborate_episode_logs(logs, discount):

    return_per_episode = synthesize(logs["return_per_episode"])
    rreturn_per_episode = synthesize(logs["reshaped_return_per_episode"])
    avg_reward_per_step = average_reward_per_step(logs["return_per_episode"], logs["num_frames_per_episode"])
    avg_discounted_return = average_discounted_return(logs["return_per_episode"], logs["num_frames_per_episode"], discount)
    num_frames_per_episode = synthesize(logs["num_frames_per_episode"])

    episode_logs = {
        "return_per_episode": return_per_episode,
        "rreturn_per_episode": rreturn_per_episode,
        "average_reward_per_step": avg_reward_per_step,
        "average_discounted_return": avg_discounted_return,
        "num_frames_per_episode": num_frames_per_episode,
        "num_frames": logs["num_frames"]
    }

    return episode_logs