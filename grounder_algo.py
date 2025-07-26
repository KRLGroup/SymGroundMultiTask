import torch
import numpy as np

import utils
from replay_buffer import ReplayBuffer
from deep_automa import MultiTaskProbabilisticAutoma


# class for training the grounder
class GrounderAlgo():

    def __init__(self, grounder, train_grounder, sampler, env, max_steps=50, batch_size=32, capacity=1000, lr=0.001, device=None):

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.env = env
        self.capacity = capacity
        self.batch_size = batch_size
        self.num_symbols = len(env.propositions)
        self.max_steps = max_steps
        self.train_grounder = train_grounder

        self.grounder = grounder
        self.sampler = sampler

        if train_grounder:
            self.buffer = ReplayBuffer(capacity=capacity, device=device)
            self.loss_func = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.grounder.parameters(), lr=lr)
            self.optimizer.zero_grad()

        else:
            self.buffer = None
            self.loss_func = None
            self.optimizer = None


    def add_episode(self, obss, rews, dfa_trans, dfa_rew):
        self.buffer.push(obss, rews, dfa_trans, dfa_rew)


    def process_experiences(self, exps):

        if not self.train_grounder:
            logs = {'buffer': 0}
            return logs

        ids = torch.stack([exps.obs.episode_id, exps.obs.env_id], dim=1)
        unique_ids = torch.unique(ids, dim=0)
        episodes = []

        # compose episodes
        for episode_id, env_id in unique_ids:
            mask = (exps.obs.episode_id == episode_id) & (exps.obs.env_id == env_id)
            episodes.append(exps[mask])

        for episode in episodes:

            obss = episode.obs.image
            rews = episode.reward.int()
            task = episode.obs.task_id[0]

            # if the rewards are all 0 there is no supervision
            if rews[-1] == 0:
                continue

            # extend shorter vectors to the max length
            if len(rews) < self.max_steps+1:
                last_rew = rews[-1]
                last_obs = obss[-1]
                extension = self.max_steps+1 - len(rews)
                rews = torch.cat([rews, last_rew.repeat(extension)])
                obss = torch.cat([obss, last_obs.repeat(extension, 1, 1, 1)])

            # cut longer vectors
            if len(rews) > self.max_steps+1:
                rews = rews[:self.max_steps+1]
                obss = obss[:self.max_steps+1]

                if rews[-1] == 0:
                    continue

            # load automata
            dfa = self.sampler.get_automaton(task)
            dfa_trans = dfa.transitions
            dfa_rew = dfa.rewards

            # add episode
            self.add_episode(obss, rews, dfa_trans, dfa_rew)

        logs = {'buffer': len(self.buffer)}

        return logs


    def collect_experiences(self, agent=None):

        if not self.train_grounder:
            logs = {'buffer': 0, 'num_frames': 0}
            return logs

        # disable grounder temporarily (for efficiency)
        if self.env.env.sym_grounder is not None and agent is None:
            env_grounder = self.env.env.sym_grounder
            self.env.env.sym_grounder = None

        # reset the environment
        obs = self.env.reset()

        # agent starts in an empty cell (never terminates in 0 actions)
        done = False
        obss = [obs['features']]
        rews = [0]

        # play the episode until termination
        while not done:
            action = agent.get_action(obs).item() if agent else self.env.action_space.sample()
            obs, rew, done, _ = self.env.step(action)
            obss.append(obs['features'])
            rews.append(rew)

        # reward obtained only at last step (if it's 0 there is no supervision)
        if rew != 0 and len(rews) <= self.max_steps+1:

            # extend shorter vectors to max length
            if len(rews) < self.max_steps+1:
                last_rew = rews[-1]
                last_obs = obss[-1]
                extension = self.max_steps+1 - len(rews)
                rews.extend([last_rew] * extension)
                obss.extend([last_obs] * extension)

            # add to the buffer
            obss = torch.tensor(np.stack(obss), device=self.device, dtype=torch.float32)
            rews = torch.tensor(rews, device=self.device, dtype=torch.int64)
            task = self.env.sampler.get_current_automaton()
            self.add_episode(obss, rews, task.transitions, task.rewards)

        # enable grounder back
        if self.env.env.sym_grounder is not None and agent is None:
            self.env.env.sym_grounder = env_grounder

        logs = {'buffer': len(self.buffer), 'num_frames': len(rews)}

        return logs


    def update_parameters(self):

        if not self.train_grounder or len(self.buffer) < self.batch_size:
            logs = {'grounder_loss': 0.0}
            return logs

        # sample from the buffer
        obss, rews, dfa_trans, dfa_rew = self.buffer.sample(self.batch_size)

        # build the differentiable reward machine for the task
        deepDFA = MultiTaskProbabilisticAutoma(
            batch_size = self.batch_size,
            numb_of_actions = self.num_symbols,
            numb_of_states = max([len(tr.keys()) for tr in dfa_trans]),
            reward_type = "ternary",
            device = self.device
        )
        deepDFA.initFromDfas(dfa_trans, dfa_rew)

        # reset gradient
        self.optimizer.zero_grad()

        # obtain probability of symbols from observations with self.grounder
        symbols = self.grounder(obss.view(-1, *obss.shape[2:]))
        symbols = symbols.view(*obss.shape[:2], -1)

        # predict state and reward from predicted symbols with DeepDFA
        pred_states, pred_rew = deepDFA(symbols)
        pred = pred_rew.view(-1, deepDFA.numb_of_rewards)

        # maps rewards to label
        labels = (rews + 1).view(-1)

        # compute loss
        loss = self.loss_func(pred, labels)

        # update self.grounder
        loss.backward()
        self.optimizer.step()

        # log some values
        logs = {'grounder_loss': loss.item()}

        return logs


    def evaluate(self):

        if not self.train_grounder:
            logs = {
                'grounder_acc': 0.0,
                'grounder_recall': [0.0 for _ in range(self.num_symbols)]
            }
            return logs

        coords = self.env.env.loc_to_label.keys()

        # obtain and preprocess data
        images = np.stack([self.env.env.loc_to_obs[(r, c)] for (r, c) in coords])
        images = torch.tensor(images, device=self.device, dtype=torch.float32)
        real_syms = [self.env.env.loc_to_label[(r, c)] for (r, c) in coords]
        real_syms = torch.tensor(real_syms, device=self.device, dtype=torch.int32)

        # predict symbols
        pred_syms = torch.argmax(self.grounder(images), dim=-1)
        correct_preds = torch.sum((pred_syms == real_syms).int())
        acc = torch.mean((pred_syms == real_syms).float())

        # compute recall
        true_pos = torch.zeros(self.num_symbols, device=self.device)
        false_neg = torch.zeros(self.num_symbols, device=self.device)
        for sym in range(self.num_symbols):
            true_pos[sym] = torch.sum((pred_syms == sym) & (real_syms == sym))
            false_neg[sym] = torch.sum((pred_syms != sym) & (real_syms == sym))
        recall = true_pos / (true_pos + false_neg + 1e-8)

        # log some values
        logs = {
            'grounder_acc': acc.item(),
            'grounder_recall': recall.tolist()
        }

        return logs