import sys
import multiprocessing as mp

from lab import test
from train_agent import train_agent


if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)
    train_agent(test.test_gridworld, device=sys.argv[1])