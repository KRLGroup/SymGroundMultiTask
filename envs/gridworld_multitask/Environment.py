import gym
from gym import spaces
import random
import numpy as np
import torch, torchvision
from FiniteStateMachine import MooreMachine
from itertools import product
import pickle
import cv2
import os
from ltl_wrappers import LTLEnv


resize = torchvision.transforms.Resize((64,64))
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    resize,
])

ENV_DIR = os.path.dirname(os.path.abspath(__file__))


class GridWorldEnv_multitask(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="human", state_type = "image", train=True, size=7, max_num_steps = 70, randomize_loc = False, img_dir="imgs"):
        
        self.dictionary_symbols = ['a', 'b', 'c', 'd', 'e', 'f']  # CHANGED FROM ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']

        self.randomize_locations = randomize_loc
        self.multitask_urs = set(product(list(range(len(self.dictionary_symbols))), repeat=len(self.dictionary_symbols)))
        self.produced_tasks = 0

        self._PICKAXE = os.path.join(ENV_DIR, img_dir, "pickaxe.png")
        self._LAVA = os.path.join(ENV_DIR, img_dir, "lava.png")
        self._DOOR = os.path.join(ENV_DIR, img_dir, "door.png")
        self._GEM = os.path.join(ENV_DIR, img_dir, "gem.png")
        self._EGG = os.path.join(ENV_DIR, img_dir, "turtle_egg.png")
        self._ROBOT = os.path.join(ENV_DIR, img_dir, "robot.png")

        self._train = train
        self.max_num_steps = max_num_steps
        self.curr_step = 0

        self.state_type = state_type
        self.size = size
        self.window_size = 896

        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.has_window = False
        self.clock = None

        # load automata and formulas
        with open(os.path.join(ENV_DIR, "tasks/formulas.pkl"), "rb") as f:
            self.formulas = pickle.load(f)
        with open(os.path.join(ENV_DIR, "tasks/automata.pkl"), "rb") as f:
            self.automata = pickle.load(f)

        for i in range(len(self.formulas)):
            new_transitions = self.automata[i].transitions
            for state in self.automata[i].transitions.keys():
                new_transitions[state][5]= state

            self.automata[i].transitions = new_transitions

        #self.current_formula = self.ltl_sampler.sample()
        '''
        print(f"current formula: {self.current_formula}")
        self.automaton = MooreMachine(self.current_formula,len(self.dictionary_symbols), f"random_task_{self.produced_tasks}", reward="acceptance", dictionary_symbols=self.dictionary_symbols)
        self.produced_tasks+=1

        self.singletask_urs, _ = find_reasoning_shortcuts(self.automaton)
        '''

        print(f"Iter {self.produced_tasks}:\t num shortcuts: {len(self.multitask_urs)}")

        #calculate the maximum reward:
        '''
        set_rew = set()
        for r in self.automaton.rewards:
            if r >= 0:
                set_rew.add(r)
        self.max_reward = sum(set_rew) - self.max_num_steps
        print("MAXIMUM REWARD:", self.max_reward)
        '''

        self.action_space = spaces.Discrete(4)
        # 0 = GO_DOWN
        # 1 = GO_RIGHT
        # 2 = GO_UP
        # 3 = GO_LEFT

        self._action_to_direction = {
            0: np.array([0, 1]),  # DOWN
            1: np.array([1, 0]),  # RIGHT
            2: np.array([0, -1]),  # UP
            3: np.array([-1, 0]),  # LEFT
        }

        self.symbol_to_meaning = {
            'a': 'pickaxe',
            'b': 'lava',
            'c': 'door',
            'd': 'gem',
            'e': 'egg',
            'f': 'nothing'
        }

        self._pickaxe_locations = [np.array([1,1]), np.array([5,2])]
        self._lava_locations = [np.array([3,3]), np.array([1,4])]
        self._door_locations = [np.array([3,0]), np.array([3,5])]
        self._gem_locations = [np.array([0,3]), np.array([6,4])]
        self._egg_locations = [np.array([2,1]), np.array([5,6])]
        self._initial_agent_location = np.array([0,0])

        # ???
        self._gem_display = True
        self._pickaxe_display = True
        self._robot_display = True

        if state_type == "image":
            #*******************************
            # Load images using OpenCV; note that OpenCV loads images as BGR by default.
            self.pickaxe_img = cv2.imread(self._PICKAXE, cv2.IMREAD_UNCHANGED)
            self.gem_img = cv2.imread(self._GEM, cv2.IMREAD_UNCHANGED)
            self.door_img = cv2.imread(self._DOOR, cv2.IMREAD_UNCHANGED)
            self.robot_img = cv2.imread(self._ROBOT, cv2.IMREAD_UNCHANGED)
            self.lava_img = cv2.imread(self._LAVA, cv2.IMREAD_UNCHANGED)
            self.egg_img = cv2.imread(self._EGG, cv2.IMREAD_UNCHANGED)
            #print(self.window_size)
            #print(self.size)
            self.pix_square_size = int(self.window_size/self.size)
            #print(self.pix_square_size)
            #*********************************
            self.image_locations = {}
            self.image_labels = {}
            for r in range(size):
                for c in range(size):
                    self._agent_location = np.array([r, c])
                    obss = self._get_obs(1)
                    obss = torch.tensor(obss.copy(), dtype=torch.float64) / 255
                    obss = torch.permute(obss, (2, 0, 1))
                    obss = resize(obss)
                    self.image_locations[r,c] = obss
                    self.image_labels[r,c] = self._current_symbol()
            #normalization
            #all_images = list(self.image_locations.values())
            #all_img_tens = torch.stack(all_images)
            #print(all_img_tens.size())
            stdev, mean = torch.std_mean(self.image_locations[0,0])

            #print(mean.size())
            #print(mean)
            #print(stdev)
            #print(stdev.sum())

            for r in range(size):
                for c in range(size):
                    #print("original: ", self.image_locations[r,c])
                    norm_img = (self.image_locations[r,c] - mean) / (stdev + 1e-10)
                    #print("normalized:", norm_img)
                    self.image_locations[r,c] = norm_img.numpy() # CONVERTED IN NUMPY

            #visualize images after normalizations
            '''
            for r in range(size):
                for c in range(size):
                    cv2.imshow("Frame", self.image_locations[r,c].permute(1, 2, 0).numpy())
                    cv2.waitKey(100)
            '''

            self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(1), shape=self.image_locations[0,0].shape, dtype=np.float32)

        self._agent_location = self._initial_agent_location


    def reset(self):
        '''
        TUTTO IL RESET
        '''

        #self.current_formula = self.ltl_sampler.sample()
        self.current_formula = self.formulas[self.produced_tasks % len(self.formulas)]
        #print(f"Current task: {self.current_formula}")
        current_automa = self.automata[self.produced_tasks % len(self.automata)]
        self.automaton = MooreMachine(current_automa.transitions,current_automa.acceptance, f"random_task_{self.produced_tasks}", reward="acceptance", dictionary_symbols=self.dictionary_symbols)
        self.produced_tasks+=1

        #self.singletask_urs, _ = find_reasoning_shortcuts(self.automaton)
        #print(f"Iter {self.produced_tasks}:\t num shortcuts: {len(self.multitask_urs)}")

        self.curr_automaton_state = 0
        self.curr_step = 0

        #reset item location
        if self.randomize_locations and self.produced_tasks % 100 == 0:
            all_positions = [(x, y) for x in range(self.size) for y in range(self.size)]

            # Seleziona casualmente 10 posizioni senza ripetizioni
            num_items = 10
            item_positions = random.sample(all_positions, num_items+1)
            self._gem_locations = [np.array(item_positions[0]), np.array(item_positions[1])]
            self._pickaxe_locations = [np.array(item_positions[2]), np.array(item_positions[3])]
            self._door_locations = [np.array(item_positions[4]), np.array(item_positions[5])]
            self._lava_locations = [np.array(item_positions[6]), np.array(item_positions[7])]
            self._egg_locations = [np.array(item_positions[8]), np.array(item_positions[9])]
            self._initial_agent_location = np.array(item_positions[10])

            # reinizialize self.image_locations and normalizations
            self.image_locations = {}
            self.image_labels = {}
            for r in range(self.size):
                for c in range(self.size):
                    self._agent_location = np.array([r, c])
                    self._render_frame()
                    obss = self._get_obs(1)
                    obss = torch.tensor(obss.copy(), dtype=torch.float64) / 255
                    obss = torch.permute(obss, (2, 0, 1))
                    obss = resize(obss)
                    self.image_locations[r,c] = obss
                    self.image_labels[r,c] = self._current_symbol()
            #normalization
            #all_images = list(self.image_locations.values())
            #all_img_tens = torch.stack(all_images)
            #print(all_img_tens.size())
            stdev, mean = torch.std_mean(self.image_locations[self._agent_location[0],self._agent_location[1]])

            #print(mean.size())
            #print(mean)
            #print(stdev)
            #print(stdev.sum())

            for r in range(self.size):
                for c in range(self.size):
                    #print("original: ", self.image_locations[r,c])
                    norm_img = (self.image_locations[r,c] - mean) / (stdev + 1e-10)
                    #print("normalized:", norm_img)
                    self.image_locations[r,c] = norm_img

        # reset the agent location
        self._agent_location = self._initial_agent_location

        if self.state_type == "symbolic":
            observation = np.array(list(self._agent_location) + [self.curr_automaton_state])
        elif self.state_type == "image":
            one_hot_dfa_state = [0 for _ in range(self.automaton.num_of_states)]
            one_hot_dfa_state[self.curr_automaton_state] = 1
            #print("one_hot_dfa_state: ", one_hot_dfa_state)
            #observation = [np.array(one_hot_dfa_state), self.image_locations[self._agent_location[0], self._agent_location[1]]] #1 FULL Img, 0 Just the square the robot is in
            observation = self.image_locations[self._agent_location[0], self._agent_location[1]]
        else:
            raise Exception("environment with state_type = {} NOT IMPLEMENTED".format(self.state_type))

        # visualize images after normalizations
        '''
        for r in range(self.size):
            for c in range(self.size):
                cv2.imshow("Frame", self.image_locations[r, c].permute(1, 2, 0).numpy())
                cv2.waitKey(100)
        '''
        return observation, self.automaton, self.image_locations, self.image_labels


    def _current_symbol(self):
        if any(np.array_equal(self._agent_location, loc) for loc in self._pickaxe_locations):
            return 0
        elif any(np.array_equal(self._agent_location, loc) for loc in self._lava_locations):
            return 1
        elif any(np.array_equal(self._agent_location, loc) for loc in self._door_locations):
            return 2
        elif any(np.array_equal(self._agent_location, loc) for loc in self._gem_locations):
            return 3
        elif any(np.array_equal(self._agent_location, loc) for loc in self._egg_locations):
            return 4
        else:
            return 5


    def step(self, action):

        reward = -1
        self.curr_step += 1
        done = False

        # MOVEMENT
        direction = self._action_to_direction[action]

        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        sym = self._current_symbol()
        #print("symbol:", sym)
        self.new_automaton_state = self.automaton.transitions[self.curr_automaton_state][sym]
        #print("state:", self.curr_automaton_state)
        #print(self.automaton.acceptance)

        #if self.automaton.acceptance[self.curr_automaton_state]:
        #    reward = 100
        #    done = True
        #if self.new_automaton_state == self.curr_automaton_state:
        #    reward = -1
        #else:
        reward = self.automaton.rewards[self.new_automaton_state]

        self.curr_automaton_state = self.new_automaton_state

        if self.state_type == "symbolic":
            observation = np.array(list(self._agent_location) + [self.curr_automaton_state])
        elif self.state_type == "image":
            one_hot_dfa_state = [0 for _ in range(self.automaton.num_of_states)]
            one_hot_dfa_state[self.curr_automaton_state] = 1
            #print("one_hot_dfa_state: ", one_hot_dfa_state)
            #observation = [np.array(one_hot_dfa_state), self.image_locations[self._agent_location[0], self._agent_location[1]]]
            observation =self.image_locations[self._agent_location[0], self._agent_location[1]]
        else:
            raise Exception("environment with state_type = {} NOT IMPLEMENTED".format(self.state_type))

        #          success            failure                  timeout
        done = (reward == 1) or (reward == -1) or (self.curr_step >= self.max_num_steps)

        # info = self._get_info()
        #if reward == 1:
        #    self.multitask_urs = self.multitask_urs.intersection(self.singletask_urs)

        return observation, reward, done


    def render(self):
        return self.image_locations[self._agent_location[0], self._agent_location[1]]


    def _get_obs(self, full = 1):
        return self._render_frame()


    def _get_info(self):
        info = {
            "robot location": self._agent_location,
            "inventory": "gem" if self._has_gem else "pickaxe" if self._has_pickaxe else "empty"
        }
        return info


    def _render_frame(self):
        # Create a white canvas.
        canvas = 255 * np.ones((self.window_size, self.window_size, 3), dtype=np.uint8)

        # Draw grid lines.
        '''
        for i in range(0, self.window_size + 1, self.pix_square_size):
            # Horizontal line
            cv2.line(canvas, (0, i), (self.window_size, i), color=(0, 0, 0), thickness=3)
            # Vertical line
            cv2.line(canvas, (i, 0), (i, self.window_size), color=(0, 0, 0), thickness=3)
        '''

        # Helper function to overlay an image with transparency if available.
        def overlay_image(bg, fg, top_left):
            x, y = top_left
            h, w = fg.shape[:2]
            # If the foreground has an alpha channel, use it for blending.
            if fg.shape[2] == 4:
                alpha_fg = fg[:, :, 3] / 255.0
                alpha_bg = 1.0 - alpha_fg
                for c in range(0, 3):
                    bg[y:y + h, x:x + w, c] = (alpha_fg * fg[:, :, c] +
                                               alpha_bg * bg[y:y + h, x:x + w, c])
            else:
                bg[y:y + h, x:x + w] = fg
            return bg

        # Calculate pixel positions for each grid item.
        def blit_item(item_img, locations, display=True):
            if display:
                for loc in locations:
                    # loc is assumed to be a numpy array [col, row] or [x, y]
                    x = int(loc[0] * self.pix_square_size)
                    y = int(loc[1] * self.pix_square_size)
                    overlay_image(canvas, item_img, (x, y))

        # Blit each type of item.
        blit_item(self.pickaxe_img, self._pickaxe_locations, self._pickaxe_display)
        blit_item(self.gem_img, self._gem_locations, self._gem_display)
        blit_item(self.door_img, self._door_locations)
        blit_item(self.lava_img, self._lava_locations)
        blit_item(self.egg_img, self._egg_locations)
        blit_item(self.robot_img, [self._agent_location], self._robot_display)

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


    def show(self):
        assert self.render_mode == "human"

        if not self.has_window:
            self.has_window = True
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Frame", self.window_size, self.window_size)

        canvas = cv2.cvtColor(self._render_frame(), cv2.COLOR_RGB2BGR)
        cv2.imshow("Frame", canvas)
        cv2.waitKey(1)


    def close(self):
        if self.has_window:
            self.has_window = False
            cv2.destroyWindow("Frame")



# interface needed by ltl2action to build the ltl_wrapper with the learned symbol grounding
class GridWorldEnv_LTL2Action(GridWorldEnv_multitask):

    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.sym_grounder = torch.load('sym_grounder.pth', map_location=self.device)
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
        img = self.current_obs
        pred_sym = torch.argmax(self.sym_grounder(torch.tensor(img, device=self.device).unsqueeze(0)), dim=-1)[0]
        return self.dictionary_symbols[pred_sym]



class LTLWrapper(LTLEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = None # make sure we don't use this


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

        reward = original_reward #+ ltl_reward
        done = env_done or ltl_done
        return ltl_obs, reward, done, info


    def sample_ltl_goal(self):
        return self.env.current_formula