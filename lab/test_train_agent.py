import sys

from lab import test
from train_agent import train_agent

train_agent(test.test_gridworld, device=sys.argv[1])
# train_agent(test.test_ltl2action, device=sys.argv[1])

