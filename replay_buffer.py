import random
from collections import deque
import torch
import numpy as np


class ReplayBuffer:

    def __init__(self, capacity=1000, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.total_episodes = 0
        self.buffer = deque(maxlen=capacity)


    def push(self, obss, rews, dfa_trans, dfa_rew):
        self.buffer.append((obss, rews, dfa_trans, dfa_rew))
        self.total_episodes += 1


    def __len__(self):
        return len(self.buffer)


    def __iter__(self):
        for obss, rews, dfa_trans, dfa_rew in self.buffer:
            yield (
                obss.to(self.device),
                rews.to(self.device),
                dfa_trans,
                dfa_rew
            )


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obss, rews, dfa_trans, dfa_rew = zip(*batch)
        obss = self._pad_repeat_last(obss).to(self.device)
        rews = self._pad_repeat_last(rews).to(self.device)
        return obss, rews, dfa_trans, dfa_rew


    def _pad_repeat_last(self, sequences):
        max_len = max(seq.shape[0] for seq in sequences)
        padded = []
        for seq in sequences:
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                repeat = seq[-1:].expand(pad_len, *seq.shape[1:])
                seq = torch.cat([seq, repeat], dim=0)
            padded.append(seq)
        return torch.stack(padded)


    def iter_batches(self, batch_size):
        buffer_list = list(self.buffer)
        for i in range(0, len(buffer_list), batch_size):
            batch = buffer_list[i:i + batch_size]
            obss, rews, dfa_trans, dfa_rew = zip(*batch)
            obss = self._pad_repeat_last(obss).to(self.device)
            rews = self._pad_repeat_last(rews).to(self.device)
            yield obss, rews, dfa_trans, dfa_rew