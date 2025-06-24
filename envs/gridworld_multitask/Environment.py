import gym
from gym import spaces
import random
import numpy as np
import torch
from itertools import product
import pickle
import cv2
import os
from ltl_wrappers import LTLEnv
from ltl_samplers import getLTLSampler


ENV_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(ENV_DIR))


class GridWorldEnv_multitask(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array", "terminal"],
        "state_types": ["image", "symbol"],
        "render_fps": 4
    }

    def __init__(self, render_mode="human", state_type="image", obs_size=(64,64), win_size=(896,896), size=7,
        max_num_steps=75, randomize_loc=False, randomize_start=True, img_dir="imgs_16x16", ltl_sampler="Dataset_e54",
        save_obs=False, wrap_around_map=True, agent_centric_view=True):

        self.dictionary_symbols = ['a', 'b', 'c', 'd', 'e', '']

        self.randomize_locations = randomize_loc
        self.randomize_start = randomize_start

        self._PICKAXE = os.path.join(ENV_DIR, img_dir, "pickaxe.png")
        self._LAVA = os.path.join(ENV_DIR, img_dir, "lava.png")
        self._DOOR = os.path.join(ENV_DIR, img_dir, "door.png")
        self._GEM = os.path.join(ENV_DIR, img_dir, "gem.png")
        self._EGG = os.path.join(ENV_DIR, img_dir, "turtle_egg.png")
        self._ROBOT = os.path.join(ENV_DIR, img_dir, "robot.png")

        self.max_num_steps = max_num_steps
        self.curr_step = 0
        self.has_window = False

        self.size = size
        self.obs_size = obs_size
        self.win_size = win_size

        assert not agent_centric_view or (self.size%2==1 and wrap_around_map)
        self.wrap_around_map = wrap_around_map
        self.agent_centric_view = agent_centric_view

        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        assert state_type in self.metadata["state_types"]
        self.state_type = state_type

        self.sampler = getLTLSampler(ltl_sampler, self.dictionary_symbols)

        # self.multitask_urs = set(product(list(range(len(self.dictionary_symbols))), repeat=len(self.dictionary_symbols)))
        # print(f"Iter {self.sampler.sampled_tasks}:\t num shortcuts: {len(self.multitask_urs)}")

        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]),  # DOWN
            1: np.array([1, 0]),  # RIGHT
            2: np.array([0, -1]),  # UP
            3: np.array([-1, 0]),  # LEFT
        }

        # default locations
        all_locations = {(x, y) for x in range(self.size) for y in range(self.size)}
        default_locations = [(1,1), (5,2), (3,3), (1,4), (3,0), (3,5), (0,3), (6,4), (2,1), (5,6)]

        # assign items locations
        self._pickaxe_locations = [default_locations[0], default_locations[1]]
        self._lava_locations = [default_locations[2], default_locations[3]]
        self._door_locations = [default_locations[4], default_locations[5]]
        self._gem_locations = [default_locations[6], default_locations[7]]
        self._egg_locations = [default_locations[8], default_locations[9]]

        # assign agent initial location
        self.free_locations = all_locations - set(default_locations)
        self._initial_agent_location = (0,0)

        # precompute symbols per location
        self.loc_to_label = {(r, c): 5 for r in range(self.size) for c in range(self.size)}
        for loc in self._pickaxe_locations:
            self.loc_to_label[loc] = 0
        for loc in self._lava_locations:
            self.loc_to_label[loc] = 1
        for loc in self._door_locations:
            self.loc_to_label[loc] = 2
        for loc in self._gem_locations:
            self.loc_to_label[loc] = 3
        for loc in self._egg_locations:
            self.loc_to_label[loc] = 4

        # ???
        self._gem_display = True
        self._pickaxe_display = True
        self._robot_display = True

        # load icons using OpenCV (if they are used)
        if self.state_type == 'image' or self.render_mode in ['human', 'rgb_array']:

            self.pickaxe_img = cv2.imread(self._PICKAXE, cv2.IMREAD_UNCHANGED)
            self.gem_img = cv2.imread(self._GEM, cv2.IMREAD_UNCHANGED)
            self.door_img = cv2.imread(self._DOOR, cv2.IMREAD_UNCHANGED)
            self.robot_img = cv2.imread(self._ROBOT, cv2.IMREAD_UNCHANGED)
            self.lava_img = cv2.imread(self._LAVA, cv2.IMREAD_UNCHANGED)
            self.egg_img = cv2.imread(self._EGG, cv2.IMREAD_UNCHANGED)

            # dimensions of true canvas
            self.cell_size = self.pickaxe_img.shape[0]
            self.canvas_size = self.size * self.cell_size

        if self.state_type == "image":

            # precompute image observations per location
            self.loc_to_obs = {}
            for r in range(self.size):
                for c in range(self.size):
                    self._agent_location = np.array([r, c])
                    self.loc_to_obs[r,c] = self._get_image_obs()

            # save images as seen by the agent
            if save_obs:
                obs_folder = os.path.join(REPO_DIR, 'saves/env_obs')
                if not os.path.exists(obs_folder):
                    os.makedirs(obs_folder)
                for r in range(self.size):
                    for c in range(self.size):
                        image = (np.transpose(self.loc_to_obs[r,c], (1, 2, 0)) * 255).astype(np.uint8)
                        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        obs_path = os.path.join(obs_folder, f'obs_{r}_{c}.png')
                        cv2.imwrite(obs_path, image_bgr)

            # normalize observations
            mean = np.mean(self.loc_to_obs[self._initial_agent_location])
            stdev = np.std(self.loc_to_obs[self._initial_agent_location])
            for r in range(self.size):
                for c in range(self.size):
                    norm_img = (self.loc_to_obs[r,c] - mean) / (stdev + 1e-10)
                    self.loc_to_obs[r,c] = norm_img

            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.loc_to_obs[0,0].shape, dtype=np.float64)

        elif self.state_type == "symbol":

            # precompute symbol observations per location
            self.loc_to_obs = {}
            for r in range(self.size):
                for c in range(self.size):
                    self._agent_location = np.array([r, c])
                    self.loc_to_obs[r,c] = self._get_symbol_obs()

            self.observation_space = spaces.Box(low=0, high=1, shape=self.loc_to_obs[0,0].shape, dtype=np.uint8)

        # reset the agent location
        self._agent_location = np.array(self._initial_agent_location)


    def reset(self):

        # extract task
        self.current_formula, self.current_automaton = self.sampler.sample()

        # the initial state is always 0
        self.curr_automaton_state = 0

        # self.singletask_urs, _ = find_reasoning_shortcuts(self.current_automaton)
        # print(f"Iter {self.sampler.sampled_tasks}:\t num shortcuts: {len(self.multitask_urs)}")

        self.curr_step = 0

        # randomize item locations and recompute observations
        if self.randomize_locations and self.sampler.sampled_tasks % 10 == 0:

            all_locations = {(x, y) for x in range(self.size) for y in range(self.size)}

            # select 10 random locations
            num_items = 10
            sampled_locations = random.sample(all_locations, num_items)

            # assign item locations
            self._gem_locations = [sampled_locations[0], sampled_locations[1]]
            self._pickaxe_locations = [sampled_locations[2], sampled_locations[3]]
            self._door_locations = [sampled_locations[4], sampled_locations[5]]
            self._lava_locations = [sampled_locations[6], sampled_locations[7]]
            self._egg_locations = [sampled_locations[8], sampled_locations[9]]

            # assign agent initial location
            self.free_locations = all_locations - set(items_locations)
            self._initial_agent_location = random.sample(self.free_locations)

            # precompute symbols per location
            self.loc_to_label = {(r, c): 5 for r in range(self.size) for c in range(self.size)}
            for loc in self._pickaxe_locations:
                self.loc_to_label[loc] = 0
            for loc in self._lava_locations:
                self.loc_to_label[loc] = 1
            for loc in self._door_locations:
                self.loc_to_label[loc] = 2
            for loc in self._gem_locations:
                self.loc_to_label[loc] = 3
            for loc in self._egg_locations:
                self.loc_to_label[loc] = 4

            if self.state_type == "image":

                # precompute image observations per location
                self.loc_to_obs = {}
                for r in range(self.size):
                    for c in range(self.size):
                        self._agent_location = np.array([r, c])
                        self.loc_to_obs[r,c] = self._get_image_obs()

                # normalize observations
                mean = np.mean(self.loc_to_obs[self._initial_agent_location])
                stdev = np.std(self.loc_to_obs[self._initial_agent_location])
                for r in range(self.size):
                    for c in range(self.size):
                        norm_img = (self.loc_to_obs[r,c] - mean) / (stdev + 1e-10)
                        self.loc_to_obs[r,c] = norm_img

            elif self.state_type == "symbol":

                # precompute symbol observations per location
                self.loc_to_obs = {}
                for r in range(self.size):
                    for c in range(self.size):
                        self._agent_location = np.array([r, c])
                        self.loc_to_obs[r,c] = self._get_symbol_obs()

        # extract new initial location
        elif self.randomize_start:
            self._initial_agent_location = random.sample(self.free_locations, 1)[0]

        # reset the agent location
        self._agent_location = np.array(self._initial_agent_location)

        # compute initial observation
        observation = self.loc_to_obs[tuple(self._agent_location)]

        return observation, self.current_automaton, self.loc_to_obs, self.loc_to_label


    def step(self, action):

        # update agent location
        direction = self._action_to_direction[action]
        if self.wrap_around_map:
            self._agent_location = (self._agent_location + direction) % self.size
        else:
            self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        self.curr_step += 1

        # update automaton
        sym = self.loc_to_label[tuple(self._agent_location)]
        self.new_automaton_state = self.current_automaton.transitions[self.curr_automaton_state][sym]
        reward = self.current_automaton.rewards[self.new_automaton_state]
        self.curr_automaton_state = self.new_automaton_state

        # compute observation
        observation = self.loc_to_obs[tuple(self._agent_location)]

        # compute completion state
        done = (reward == 1) or (reward == -1) or (self.curr_step >= self.max_num_steps)

        # info = self._get_info()

        # if reward == 1:
        #    self.multitask_urs = self.multitask_urs.intersection(self.singletask_urs)

        return observation, reward, done


    def _get_symbol_obs(self):
        obs = np.zeros(shape=(self.size,self.size,len(self.dictionary_symbols)+1),dtype=np.uint8)
        for loc in self.loc_to_label:
            if self.agent_centric_view:
                loc = self._absolute_to_agent_centric(loc)
            label = self.loc_to_label[loc]
            obs[loc[0],loc[1],label] = 1
        obs[self._agent_location[0],self._agent_location[1],len(self.dictionary_symbols)] = 1
        return obs


    def _get_image_obs(self):
        obs = self._render_frame()
        obs = cv2.resize(obs, self.obs_size)
        obs = obs.astype(np.float64) / 255.0
        obs = np.transpose(obs, (2, 0, 1)) # from w*h*c to c*w*h
        return obs


    def _get_info(self):
        info = {
            "robot location": self._agent_location,
            "inventory": "gem" if self._has_gem else "pickaxe" if self._has_pickaxe else "empty"
        }
        return info


    # create the visualization of the environment (for rendering and for observations)
    def _render_frame(self, draw_grid=False):

        # create a white canvas
        canvas = 255 * np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)

        # draw grid lines
        if draw_grid:
            for i in range(0, self.canvas_size + 1, self.cell_size):
                cv2.line(canvas, (0, i), (self.canvas_size, i), color=(0, 0, 0), thickness=3)
                cv2.line(canvas, (i, 0), (i, self.canvas_size), color=(0, 0, 0), thickness=3)

        # helper function to overlay an image with transparency if available
        def overlay_image(bg, fg, top_left):
            x, y = top_left
            h, w = fg.shape[:2]
            # if the foreground has an alpha channel, use it for blending
            if fg.shape[2] == 4:
                alpha_fg = fg[:, :, 3] / 255.0
                alpha_bg = 1.0 - alpha_fg
                for c in range(0, 3):
                    bg[y:y + h, x:x + w, c] = (alpha_fg * fg[:, :, c] + alpha_bg * bg[y:y + h, x:x + w, c])
            else:
                bg[y:y + h, x:x + w] = fg
            return bg

        # calculate pixel positions for each grid item
        def blit_item(item_img, locations, display=True):
            if display:
                for loc in locations:
                    # loc is assumed to be a tuple
                    if self.agent_centric_view:
                        loc = self._absolute_to_agent_centric(loc)
                    x = int(loc[0] * self.cell_size)
                    y = int(loc[1] * self.cell_size)
                    overlay_image(canvas, item_img, (x, y))

        # blit each type of item
        blit_item(self.pickaxe_img, self._pickaxe_locations, self._pickaxe_display)
        blit_item(self.gem_img, self._gem_locations, self._gem_display)
        blit_item(self.door_img, self._door_locations)
        blit_item(self.lava_img, self._lava_locations)
        blit_item(self.egg_img, self._egg_locations)
        blit_item(self.robot_img, [self._agent_location], self._robot_display)

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


    def _absolute_to_agent_centric(self, pos):
        center = self.size // 2
        delta  = (center - self._agent_location[0], center - self._agent_location[1])
        new_pos_r = (pos[0] + delta[0] + self.size) % self.size
        new_pos_c = (pos[1] + delta[1] + self.size) % self.size
        return (new_pos_r, new_pos_c)


    def _agent_centric_to_absolute(self, pos):
        center = self.size // 2
        delta  = (center - self._agent_location[0], center - self._agent_location[1])
        orig_r = (pos[0] - delta[0] + self.size) % self.size
        orig_c = (pos[1] - delta[1] + self.size) % self.size
        return (orig_r, orig_c)


    def render(self):
        if self.render_mode == "human":
            return self.show()
        elif self.render_mode == "rgb_array":
            return self.loc_to_obs[tuple(self._agent_location)]
        elif self.render_mode == "terminal":
            return self.show_to_terminal()


    def translate_formula(self, formula):

        symbol_to_meaning = {
            'a': 'pickaxe',
            'b': 'lava',
            'c': 'door',
            'd': 'gem',
            'e': 'egg',
            '': 'nothing'
        }

        if isinstance(formula, tuple):
            return tuple(self.translate_formula(item) for item in formula)
        elif formula in symbol_to_meaning:
            return symbol_to_meaning[formula]
        else:
            return formula


    def show_to_terminal(self):

        label_to_icon = {
            0: 'P',
            1: 'L',
            2: 'D',
            3: 'G',
            4: 'E',
            5: '.',
        }

        for c in range(self.size):
            row_str = ""
            for r in range(self.size):
                loc = (r, c)
                if self.agent_centric_view:
                    loc = self._agent_centric_to_absolute(loc)
                label = self.loc_to_label.get(loc, 5)
                symbol = label_to_icon.get(label, '?')
                if tuple(self._agent_location) == loc:
                    row_str += f"[{symbol}]"
                else:
                    row_str += f" {symbol} "
            print(row_str)


    def show(self):
        if not self.has_window:
            self.has_window = True
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Frame", self.win_size[0], self.win_size[1])
            cv2.moveWindow('Frame', 100, 100)
        canvas = cv2.cvtColor(self._render_frame(), cv2.COLOR_RGB2BGR)
        canvas = cv2.resize(canvas, self.win_size, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Frame", canvas)
        cv2.waitKey(1)


    def close(self):
        if self.has_window:
            self.has_window = False
            cv2.destroyWindow("Frame")



# interface needed by ltl2action to build the ltl_wrapper
# incorporates the symbol grounder
class GridWorldEnv_LTL2Action(GridWorldEnv_multitask):

    def __init__(self, grounder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sym_grounder = grounder
        self.current_obs = None


    def reset(self):
        obs, _, _, _ = super().reset()
        self.current_obs = obs
        return obs


    def step(self, action):
        obs, rew, done = super().step(action)
        self.current_obs = obs
        return obs, rew, done, {}


    def get_propositions(self):
        return self.dictionary_symbols.copy()


    def get_events(self):

        # returns the proposition that currently holds
        if self.sym_grounder == None:
            true_sym = self.loc_to_label[tuple(self._agent_location)]
            return self.dictionary_symbols[true_sym]

        # returns the proposition that currently holds according to the grounder
        else:
            img = torch.tensor(self.current_obs, device=self.sym_grounder.device).unsqueeze(0)
            pred_sym = torch.argmax(self.sym_grounder(img), dim=-1)[0]
            return self.dictionary_symbols[pred_sym]



class GridWorldEnv_Base(GridWorldEnv_LTL2Action):
    def __init__(self, grounder, obs_size):
        super().__init__(
            grounder = grounder,
            obs_size = obs_size,
            randomize_loc = False,
            wrap_around_map = True,
            agent_centric_view = False
        )



class GridWorldEnv_Agent_Centric(GridWorldEnv_LTL2Action):
    def __init__(self, grounder, obs_size):
        super().__init__(
            grounder = grounder,
            obs_size = obs_size,
            randomize_loc = False,
            wrap_around_map = True,
            agent_centric_view = True
        )



class GridWorldEnv_No_Wrap_Around(GridWorldEnv_LTL2Action):
    def __init__(self, grounder, obs_size):
        super().__init__(
            grounder = grounder,
            obs_size = obs_size,
            randomize_loc = False,
            wrap_around_map = False,
            agent_centric_view = False
        )



class LTLWrapper(LTLEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env.sampler = self.sampler


    def step(self, action):
        int_reward = 0
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)

        # progressing the ltl formula
        truth_assignment = self.get_events(self.obs, action, next_obs)
        self.ltl_goal = self.progression(self.ltl_goal, truth_assignment)
        self.obs = next_obs

        # Computing the LTL reward and done signal
        ltl_reward = 0.0
        ltl_done = False
        if self.ltl_goal == 'True':
            ltl_reward = 1.0
            ltl_done = True
        elif self.ltl_goal == 'False':
            ltl_reward = -1.0
            ltl_done = True
        else:
            ltl_reward = int_reward

        # Computing the new observation and returning the outcome of this action
        if self.progression_mode == "full":
            ltl_obs = {'features': self.obs,'text': self.ltl_goal}
        elif self.progression_mode == "none":
            ltl_obs = {'features': self.obs,'text': self.ltl_original}
        elif self.progression_mode == "partial":
            ltl_obs = {'features': self.obs, 'progress_info': self.progress_info(self.ltl_goal)}
        else:
            raise NotImplementedError

        reward = original_reward # + ltl_reward [not used?]
        done = env_done or ltl_done
        return ltl_obs, reward, done, info


    # the formula is set by the environment
    def sample_ltl_goal(self):
        return self.env.current_formula