import sys

from lab import test
from train_agent import train_agent

train_agent(test.test_simple_ltl_5l, device=sys.argv[1])