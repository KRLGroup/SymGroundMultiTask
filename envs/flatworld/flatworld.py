# Code adapted from https://github.com/clvoloshin/RL-LTL/blob/main/envs/base_envs/flatworld.py
# and https://github.com/clvoloshin/RL-LTL/blob/main/envs/base_envs/flatworld.py

from dataclasses import dataclass
from typing import Any

import numpy as np
import gym
from gym import spaces
import torch

import matplotlib.pyplot as plt

import cv2


@dataclass
class Circle:
    center: np.ndarray
    radius: float
    color: str
    symbol: str


class FlatWorld(gym.Env):

    symbol_to_color = {
        'a':'red', 'b':'blue', 'c': 'green',
        'd': 'yellow', 'e': 'aqua', 'f': 'magenta'
    }

    color_to_rgb = {
        'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0), 'yellow': (255, 255, 0),
        'aqua': (0, 255, 255), 'magenta': (255, 0, 255), 'black': (0, 0, 0)
    }

    def __init__(self, radius=0.7, delta_t=0.08, obs_size=(56,56), win_size=(896,896), max_num_steps=75,
        symbols=['a','b','c','d','e'], use_continuous_actions=True):

        self.dictionary_symbols = symbols + ['']
        self.num_symbols = len(self.dictionary_symbols)
        self.max_num_steps = max_num_steps

        self.curr_step = 0
        self.num_episodes = 0

        self.has_window = False
        self.obs_size = obs_size
        self.win_size = win_size

        self.use_continuous_actions = use_continuous_actions
        self.radius = radius
        self.delta_t = delta_t

        self.action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0]),
            4: np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]),
            5: np.array([1 / np.sqrt(2), -1 / np.sqrt(2)]),
            6: np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)]),
            7: np.array([-1 / np.sqrt(2), -1 / np.sqrt(2)]),
            8: np.array([0, 0])
        }

        self.observation_space = spaces.Box(
            low = np.float32(-np.inf),
            high = np.float32(np.inf),
            shape = (3, 56, 56),
            dtype = np.float32
        )

        if self.use_continuous_actions:
            self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float64)
        else:
            self.action_space = spaces.Discrete(9)

        default_locations = [
            np.array([-1.5, 1.5]),
            np.array([-1.5, 0.0]),
            np.array([-1.5, -1.5]),
            np.array([0.0, 1.5]),
            np.array([0.0, 0.0]),
            np.array([0.0, -1.5]),
            np.array([1.5, 1.5]),
            np.array([1.5, 0.0]),
            np.array([1.5, -1.5])
        ]

        self.circles = []
        for i in range(self.num_symbols-1):
            self.circles.append(Circle(
                center=default_locations[i],
                radius=self.radius,
                color=self.symbol_to_color[self.dictionary_symbols[i]],
                symbol=self.dictionary_symbols[i]
            ))

        self.agent_location = np.array([-1, -1])


    def reset(self, seed=None):

        self.num_episodes += 1
        self.curr_step = 0

        self.circles = []
        for i in range(self.num_symbols-1):
            pos = self._get_disjoint_position(self.radius)
            self.circles.append(Circle(
                center=pos,
                radius=self.radius,
                color=self.symbol_to_color[self.dictionary_symbols[i]],
                symbol=self.dictionary_symbols[i]
            ))

        self.agent_location = self._get_disjoint_position(0)
        obs = self.get_obs()

        return obs


    def step(self, action):

        self.curr_step += 1

        # find direction
        if not self.use_continuous_actions:
            action = self.action_to_direction[action]
        action = np.clip(action, -1, 1)

        # apply action
        self.agent_location = self.agent_location + action.flatten() * self.delta_t
        self.agent_location = np.clip(self.agent_location, -2, 2)  # doesn't lose if hit the wall

        obs = self.get_obs()
        reward = 0.0
        done = self.curr_step >= self.max_num_steps
        info = None

        return obs, reward, done, None


    def get_active_proposition(self):
        props = []
        for circle in self.circles:
            if np.linalg.norm(self.agent_location - circle.center) < circle.radius:
                props.append(circle.symbol)
        if len(props) > 1:
            raise ValueError("Agent is in multiple circles!")
        elif len(props) == 0:
            return self.dictionary_symbols[-1]
        else:
            return props[0]


    def get_obs(self):
        obs = self._render_frame(self.obs_size)
        obs = obs.astype(np.float32) / 255.0
        obs = np.transpose(obs, (2, 0, 1)) # from w*h*c to c*w*h
        return obs


    def translate_formula(self, formula):
        if isinstance(formula, tuple):
            return tuple(self.translate_formula(item) for item in formula)
        elif formula in self.symbol_to_color:
            return self.symbol_to_color[formula]
        else:
            return formula


    def render(self, trajectory=None, ax=None):

        def hide_ticks(axis):
            for tick in axis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

        if trajectory is None:
            trajectory = []
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        for circle in self.circles:
            xy = (float(circle.center[0]), float(circle.center[1]))
            patch = plt.Circle(xy, circle.radius, color=circle.color, fill=True, alpha=.2)
            ax.add_patch(patch)

        if len(trajectory) > 0:
            trajectory = np.array(trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color='green', marker='o',
                    linestyle='dashed',
                    linewidth=2, markersize=1)
            ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], s=100, marker='o', c="orange")
            ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], s=100, marker='o', c="g")
        ax.axis('square')
        hide_ticks(ax.xaxis)
        hide_ticks(ax.yaxis)
        ax.set_xlim([-2.1, 2.1])
        ax.set_ylim([-2.1, 2.1])


    def set_pos(self, pos):
        pos = np.clip(np.array(pos), -2, 2)
        self.agent_location = pos


    def get_random_pos(self):
        return np.random.uniform(low=-2.0, high=2.0, size=(2,))


    def _get_disjoint_position(self, radius):
        pos = np.random.uniform(low=-2.0, high=2.0, size=(2,))
        while self._check_disjoint_position(pos, radius) is False:
            pos = np.random.uniform(low=-2.0, high=2.0, size=(2,))
        return pos


    def _check_disjoint_position(self, pos, radius):
        disjoint = True
        for circle in self.circles:
            if np.linalg.norm(pos - circle.center) <= (circle.radius + radius):
                disjoint = False
                break
        return disjoint


    def _render_frame(self, canvas_size):

        canvas = np.full((canvas_size[0], canvas_size[1], 3), 255, dtype=np.uint8)

        def world_to_pixel_pos(x, y, size):
            px = ((x + 2) * size / 4)
            py = ((2 - y) * size / 4)
            return int(px), int(py)

        def world_to_pixel_length(length, size):
            return int(length * (size / 4))

        def draw_circle(center, radius, color):
            px, py = world_to_pixel_pos(center[0], center[1], canvas_size[0])
            p_radius = world_to_pixel_length(radius, canvas_size[0])
            color = self.color_to_rgb[color]
            overlay = canvas.copy()
            cv2.circle(overlay, (px, py), p_radius, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
            cv2.circle(canvas, (px, py), p_radius, color, 2, cv2.LINE_AA)

        for circle in self.circles:
            draw_circle(circle.center, circle.radius, circle.color)
        draw_circle(self.agent_location, 0.1, 'black')

        return canvas


    def show(self):
        if not self.has_window:
            self.has_window = True
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Frame", self.win_size[0], self.win_size[1])
            cv2.moveWindow('Frame', 100, 100)
        canvas = cv2.cvtColor(self._render_frame(self.win_size), cv2.COLOR_RGB2BGR)
        cv2.imshow("Frame", canvas)
        cv2.waitKey(1)


    def close(self):
        if self.has_window:
            self.has_window = False
            cv2.destroyWindow("Frame")



class FlatWorld_LTL2Action(FlatWorld):

    def __init__(self, grounder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sym_grounder = grounder
        self.current_obs = None


    def reset(self):
        obs = super().reset()
        self.current_obs = obs
        return obs


    def step(self, action):
        obs, rew, done, info = super().step(action)
        self.current_obs = obs
        return obs, rew, done, info


    def get_propositions(self):
        return self.dictionary_symbols[:-1].copy()


    def get_real_events(self):
        return self.get_active_proposition()


    def get_events(self):

        # returns the proposition that currently holds
        if self.sym_grounder == None:
            return self.get_real_events()

        # returns the proposition that currently holds according to the grounder
        else:
            with torch.no_grad():
                img = torch.tensor(self.current_obs, device=self.sym_grounder.device).unsqueeze(0)
                pred_sym = torch.argmax(self.sym_grounder(img), dim=-1)[0]
            return self.dictionary_symbols[pred_sym]



# Preconstructed Environments

class FlatWorldEnv_Base(FlatWorld_LTL2Action):
    def __init__(self, state_type, grounder, obs_size, max_num_steps):
        super().__init__(
            grounder = grounder,
            obs_size = obs_size,
            max_num_steps = max_num_steps,
            symbols = ['a', 'b', 'c', 'd', 'e'],
            radius = 0.7,
            delta_t = 0.08
        )



if __name__ == '__main__':
    env = FlatWorld(use_continuous_actions=False)
    obs = env.reset()
    env.show()
    env.render()
    plt.show()