import numpy as np
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv



class CompositionalPPOAlgo(BaseAlgo):

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):

        assert acmodel.compositional

        num_frames_per_proc = num_frames_per_proc or 128

        # formulas must be updated in sync
        for env in envs:
            env.sample_on_reset = False

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.act_shape = envs[0].action_space.shape

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0


    def collect_experiences(self):

        goal, idx = self.env.envs[0].sample_ltl_goal()
        self.obs = self.env.set_goal(goal, idx)
        self.acmodel.update_formula(goal)

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)
        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

        for i in range(self.num_frames_per_proc):

            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                dist, value = self.acmodel(preprocessed_obs, self.mask)
            action = dist.sample()
            obs, reward, done, info = self.env.step(action.cpu().numpy())

            # Collect values
            self.obss[i] = self.obs
            self.last_obss[i] = [proc_info["last_obs"] for proc_info in info]
            self.obs = obs
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for j, done_ in enumerate(done):

                if done_:

                    # update
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[j].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[j].item())
                    self.log_num_frames.append(self.log_episode_num_frames[j].item())

                    # reset hidden states
                    for k, _ in enumerate(self.acmodel.base.hidden_states):
                        hidden_size = self.acmodel.base.hidden_states[k][self.acmodel.base.args.rnn_depth-1][j].shape
                        self.acmodel.base.hidden_states[k][self.acmodel.base.args.rnn_depth-1][j] = torch.zeros(hidden_size)

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Compute advantage of the state after the last step
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            _, next_value = self.acmodel(preprocessed_obs, self.mask)
            next_value = next_value

        # force all episodes to terminate (the formula will change)
        for j in range(len(done)):
            self.last_obss[-1][j] = True

        # Compute advantage
        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0
            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        # Build experience dictionary
        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        exps.last_obs = [self.last_obss[i][j]
                         for j in range(self.num_procs)
                         for i in range(self.num_frames_per_proc)]

        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape((-1, ) + self.action_space_shape)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape((-1, ) + self.action_space_shape)

        # Preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        exps.last_obs = np.array(exps.last_obs, dtype=object)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs


    def update_parameters(self, exps):

        for _ in range(self.epochs):

            # Initialize log values
            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            values = []
            entropies =  []
            log_probs = []

            # Initialize batch values
            batch_entropy = 0
            batch_value = 0
            batch_policy_loss = 0
            batch_value_loss = 0
            batch_loss = 0

            order = [i+j*self.num_frames_per_proc for i in range(self.num_frames_per_proc) for j in range(self.num_procs)]
            exps = exps[order]

            self.acmodel.reset()

            for i in range(self.num_frames_per_proc):

                sb = exps[i*self.num_procs:(i+1)*self.num_procs]

                # Compute loss components
                dist, value = self.acmodel(sb.obs, sb.mask)

                values.append(value)
                entropies.append(dist.entropy())
                log_probs.append(dist.log_probs(sb.action))

            values = torch.cat(values)
            entropies = torch.cat(entropies)
            log_probs = torch.cat(log_probs)

            # Entropy (S)
            entropy = entropies.mean()

            # Clipped policy loss (L_clip)
            delta_log_prob = log_probs - exps.log_prob
            if (len(self.act_shape) == 1): # Not scalar actions (multivariate)
                delta_log_prob = torch.sum(delta_log_prob, dim=1)
            ratio = torch.exp(delta_log_prob)
            surr1 = ratio * exps.advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * exps.advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (L_vf)
            value_clipped = exps.value + torch.clamp(values - exps.value, -self.clip_eps, self.clip_eps)
            surr1 = (values - exps.returnn).pow(2)
            surr2 = (value_clipped - exps.returnn).pow(2)
            value_loss = torch.max(surr1, surr2).mean()

            # Total loss (L_clip+vf+s)
            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

            # Update batch values
            batch_entropy += entropy.item()
            batch_value += values.mean().item()
            batch_policy_loss += policy_loss.item()
            batch_value_loss += value_loss.item()
            batch_loss += loss

            # Update actor-critic
            self.optimizer.zero_grad()
            batch_loss.backward()

            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters() if (p.requires_grad and p.grad is not None)) ** 0.5
            torch.nn.utils.clip_grad_norm_([p for p in self.acmodel.parameters() if p.requires_grad], self.max_grad_norm)
            self.optimizer.step()

            # Update log values
            log_entropies.append(batch_entropy)
            log_values.append(batch_value)
            log_policy_losses.append(batch_policy_loss)
            log_value_losses.append(batch_value_loss)
            log_grad_norms.append(grad_norm)

        logs = {
            "entropy": np.mean(log_entropies),
            "value": np.mean(log_values),
            "policy_loss": np.mean(log_policy_losses),
            "value_loss": np.mean(log_value_losses),
            "grad_norm": np.mean(log_grad_norms)
        }

        return logs