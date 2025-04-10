import gym
from gym import spaces
import pygame
import random
import numpy as np
import torch, torchvision
from FiniteStateMachine import MooreMachine
# from formula_sampling import EventuallySampler
from itertools import product
from UnremovableReasoningShurtcuts import find_reasoning_shortcuts
import pickle

resize = torchvision.transforms.Resize((64,64))
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    resize,
])
# tutta la griglia
class GridWorldEnv_multitask(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="human", state_type = "symbolic", train=True, size=7):
        self.dictionary_symbols = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5' ]
        # self.ltl_sampler = EventuallySampler(self.dictionary_symbols[:-1])
        self.multitask_urs = set(product(list(range(len(self.dictionary_symbols))), repeat=len(self.dictionary_symbols)))
        self.produced_tasks = 0

        self._PICKAXE = "imgs/pickaxe.png"
        self._GEM = "imgs/gem.png"
        self._DOOR = "imgs/door.png"
        self._ROBOT = "imgs/robot.png"
        self._LAVA = "imgs/lava.png"
        self._EGG = "imgs/turtle_egg.png"

        self._train = train
        self.max_num_steps = 70
        self.curr_step = 0

        self.state_type = state_type
        self.size = size  # 4x4 world
        self.window_size = 896  # size of the window

        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        #load automata and formulas
        with open("formulas.pkl", "rb") as f:
            self.formulas = pickle.load(f)
        with open("automata.pkl", "rb") as f:
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

        self._gem_location = [np.array([0, 3]), np.array([6,4])]
        self._pickaxe_location =[ np.array([1, 1]), np.array([5,2])]
        self._exit_location =[ np.array([3, 0]), np.array([3,5])]
        self._lava_location =[ np.array([3, 3]) , np.array([1,4])]
        self._egg_location = [np.array([2,1]), np.array([5,6])]

        self._gem_display = True
        self._pickaxe_display = True
        self._robot_display = True

        if state_type == "image":
            self.image_locations = {}
            self.image_labels = {}
            for r in range(size):
                for c in range(size):
                    self._agent_location = np.array([r, c])
                    self._render_frame()
                    obss = self._get_obs(1)
                    obss = torch.tensor(obss.copy(), dtype=torch.float64) / 255
                    obss = torch.permute(obss, (2, 0, 1))
                    obss = resize(obss)
                    self.image_locations[r,c] = obss
                    self.image_labels[r,c] = self._current_symbol()
            #normalization
            all_images = list(self.image_locations.values())
            all_img_tens = torch.stack(all_images)
            #print(all_img_tens.size())
            stdev, mean = torch.std_mean(all_img_tens, dim=0)
            #print(mean)
            #print(stdev)
            #print(stdev.sum())
            for r in range(size):
                for c in range(size):
                    norm_img = (self.image_locations[r,c] - mean) / (stdev + 1e-5)
                    #print(norm_img)
                    self.image_locations[r,c] = norm_img
            #assert False


    def reset(self):
        '''
        TUTTO IL RESET
        '''
        #reset the task
        #self.current_formula = self.ltl_sampler.sample()
        self.current_formula = self.formulas[self.produced_tasks]
        #print(f"Current task: {self.current_formula}")
        current_automa = self.automata[self.produced_tasks]
        self.automaton = MooreMachine(current_automa.transitions,current_automa.acceptance, f"random_task_{self.produced_tasks}", reward="acceptance", dictionary_symbols=self.dictionary_symbols)
        self.produced_tasks+=1

        #self.singletask_urs, _ = find_reasoning_shortcuts(self.automaton)
        #print(f"Iter {self.produced_tasks}:\t num shortcuts: {len(self.multitask_urs)}")


        self.curr_automaton_state = 0
        self.curr_step = 0

        #reset item locations
        '''
        all_positions = [(x, y) for x in range(self.size) for y in range(self.size)]

        # Seleziona casualmente 10 posizioni senza ripetizioni
        num_items = 10
        item_positions = random.sample(all_positions, num_items+1)

        self._gem_location = [np.array(item_positions[0]), np.array(item_positions[1])]
        self._pickaxe_location =[ np.array(item_positions[2]), np.array(item_positions[3])]
        self._exit_location =[ np.array(item_positions[4]), np.array(item_positions[5])]
        self._lava_location =[ np.array(item_positions[6]) , np.array(item_positions[7])]
        self._egg_location = [np.array(item_positions[8]), np.array(item_positions[9])]

        #reset the agent location
        self._agent_location = np.array(item_positions[10])
        '''
        self._agent_location = np.array([0, 0])

        #if self.render_mode == "human":
        #    self._render_frame()
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

        return observation, self.automaton

    def _current_symbol(self):
        if any(np.array_equal(self._agent_location, loc) for loc in self._exit_location):
            return 2
        if any(np.array_equal(self._agent_location, loc) for loc in self._pickaxe_location):
            return 0
        if any(np.array_equal(self._agent_location, loc) for loc in self._gem_location):
            return 3
        if any(np.array_equal(self._agent_location, loc) for loc in self._lava_location):
            return 1
        if any(np.array_equal(self._agent_location, loc) for loc in self._egg_location):
            return 4
        return 5

    def step(self, action):

        reward = -1
        self.curr_step += 1
        done = False

        # MOVEMENT
        if action == 0:
            direction = np.array([0, 1])
        elif action == 1:
            direction = np.array([1, 0])
        elif action == 2:
            direction = np.array([0, -1])
        elif action == 3:
            direction = np.array([-1, 0])

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

        #if self.render_mode == "human":
        #    self._render_frame()

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
        if self.render_mode == "rgb_array":
            return self._render_frame()
        else:
            return self.image_locations[self._agent_location[0], self._agent_location[1]]

    def _get_obs(self, full = 1):
        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )
        img = img[:, :, ::-1]
        obs = None
        if full == 1:
            obs = img
        else: 
            pix_square_size = (self.window_size/self.size)
            pix_square_size = int(pix_square_size)
            x = self._agent_location[0]
            y = self._agent_location[1]
            obs = img[int(y*pix_square_size):int((y+1)*pix_square_size), int(x*pix_square_size):int((x+1)*pix_square_size)]
        return obs

    def _get_info(self):
        info = {
            "robot location": self._agent_location,
            "inventory": "empty"
        }
        if self._has_gem:
            info["inventory"] = "gem"
        elif self._has_pickaxe:
            info["inventory"] = "pickaxe"
        else:
            info["inventory"] = "empty"
        return info

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = (self.window_size / self.size)

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            pickaxe = pygame.image.load(self._PICKAXE)
            gem = pygame.image.load(self._GEM)
            door = pygame.image.load(self._DOOR)
            robot = pygame.image.load(self._ROBOT)
            lava = pygame.image.load(self._LAVA)
            egg = pygame.image.load(self._EGG)
            self.window.blit(canvas, canvas.get_rect())

            if self._pickaxe_display:
                for pickaxe_loc in self._pickaxe_location:
                    self.window.blit(pickaxe, (pix_square_size * pickaxe_loc[0], pix_square_size * pickaxe_loc[1]))
            if self._gem_display:
                for gem_loc in self._gem_location:
                    self.window.blit(gem, (pix_square_size * gem_loc[0], pix_square_size * gem_loc[1]))
            for door_loc in self._exit_location:
                self.window.blit(door, (pix_square_size * door_loc[0], pix_square_size * door_loc[1]))
            for lava_loc in self._lava_location:
                self.window.blit(lava, (pix_square_size * lava_loc[0] , pix_square_size * lava_loc[1]))
            for egg_loc in self._egg_location:
                self.window.blit(egg, (pix_square_size * egg_loc[0], pix_square_size * egg_loc[1]))

            if self._robot_display:
                self.window.blit(robot,
                                 (pix_square_size * self._agent_location[0], pix_square_size * self._agent_location[1]))

            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()