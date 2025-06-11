import random
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


def get_dataset_duplicates(dataset_path):

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    unique = set(dataset)
    duplicate = [item for item in dataset if item not in unique]
    return duplicate


def get_dataset_common_index_pairs(dataset1_path, dataset2_path):

    with open(dataset1_path, "rb") as f:
        dataset1 = pickle.load(f)
    with open(dataset2_path, "rb") as f:
        dataset2 = pickle.load(f)

    # Find common elements
    common_elements = set(dataset1) & set(dataset2)
    
    # Build element-to-index mappings
    index_map1 = {}
    index_map2 = {}

    for i, item in enumerate(dataset1):
        if item in common_elements:
            index_map1.setdefault(item, []).append(i)

    for j, item in enumerate(dataset2):
        if item in common_elements:
            index_map2.setdefault(item, []).append(j)

    # Create index pairs for matching elements
    index_pairs = []
    for elem in common_elements:
        for i in index_map1[elem]:
            for j in index_map2[elem]:
                index_pairs.append((i, j))

    return index_pairs