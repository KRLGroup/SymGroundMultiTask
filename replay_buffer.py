import random
from collections import deque
import torch
import numpy as np


class ReplayBuffer:

    def __init__(self, capacity=1000, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.buffer = deque(maxlen=capacity)


    def push(self, obss, rews, dfa_trans, dfa_rew):
        self.buffer.append((obss, rews, dfa_trans, dfa_rew))


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obss, rews, dfa_trans, dfa_rew = zip(*batch)
        obss = torch.stack(obss).to(self.device)
        rews = torch.stack(rews).to(self.device)
        return obss, rews, dfa_trans, dfa_rew


    def __len__(self):
        return len(self.buffer)