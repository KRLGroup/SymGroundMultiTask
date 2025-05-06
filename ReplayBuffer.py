import random
from collections import deque
import torch
import numpy as np
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class ReplayBuffer:
    def __init__(self, capacity= 1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obss, rews, dfa_trans, dfa_rew):
        # Store a transition in the buffer
        self.buffer.append((obss, rews, dfa_trans, dfa_rew))

    def sample(self, batch_size):
        # Sample a random batch of transitions
        batch = random.sample(self.buffer, batch_size)
        #obss, revs, dfa_trans, dfa_rew = map(np.array, zip(*batch))
        obss, rews, dfa_trans, dfa_rew = zip(*batch)
        obss = torch.stack(obss).to(device)
        rews = torch.stack(rews).to(device)
        return obss, rews, dfa_trans, dfa_rew

    def __len__(self):
        return len(self.buffer)