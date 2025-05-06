import gym
from gym import spaces
import pygame
import random
import numpy as np
import torch, torchvision
from FiniteStateMachine import MooreMachine
from itertools import product
import pickle
import cv2

resize = torchvision.transforms.Resize((64,64))
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    resize,
])
# tutta la griglia
class GridWorldEnv_multitask(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="human", state_type = "symbolic", train=True, size=7, max_num_steps = 70, reset_loc = False, img_dir="imgs"):
        self.dictionary_symbols = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5' ]
        self.reset_locations = reset_loc
        self.multitask_urs = set(product(list(range(len(self.dictionary_symbols))), repeat=len(self.dictionary_symbols)))
        self.produced_tasks = 0

        self._PICKAXE = img_dir+"/pickaxe.png"
        self._GEM = img_dir+"/gem.png"
        self._DOOR = img_dir+"/door.png"
        self._ROBOT = img_dir+"/robot.png"
        self._LAVA = img_dir+"/lava.png"
        self._EGG = img_dir+"/turtle_egg.png"

        self._train = train
        self.max_num_steps = max_num_steps
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
        self._initial_agent_location = np.array([0,0])
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
            print(self.window_size)
            print(self.size)
            self.pix_square_size = int(self.window_size/self.size)
            print(self.pix_square_size)
            #cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("Frame", self.window_size, self.window_size)
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
                    self.image_locations[r,c] = norm_img

            #visualize images after normalizations
            '''
            for r in range(size):
                for c in range(size):
                    cv2.imshow("Frame", self.image_locations[r,c].permute(1, 2, 0).numpy())
                    cv2.waitKey(100)
            '''
        self._agent_location = self._initial_agent_location


    def reset(self):
        '''
        TUTTO IL RESET
        '''
        #reset the task
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
        if self.reset_locations:
            all_positions = [(x, y) for x in range(self.size) for y in range(self.size)]

            # Seleziona casualmente 10 posizioni senza ripetizioni
            num_items = 10
            item_positions = random.sample(all_positions, num_items+1)
            if self.produced_tasks % 10 == 0:
                self._gem_location = [np.array(item_positions[0]), np.array(item_positions[1])]
                self._pickaxe_location =[ np.array(item_positions[2]), np.array(item_positions[3])]
                self._exit_location =[ np.array(item_positions[4]), np.array(item_positions[5])]
                self._lava_location =[ np.array(item_positions[6]) , np.array(item_positions[7])]
                self._egg_location = [np.array(item_positions[8]), np.array(item_positions[9])]
                self._initial_agent_location = np.array(item_positions[10])

            #reinizialize self.image_locations and normalizations
            if self.produced_tasks % 10 == 0:

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

        # visualize images after normalizations
        '''
        for r in range(self.size):
            for c in range(self.size):
                cv2.imshow("Frame", self.image_locations[r, c].permute(1, 2, 0).numpy())
                cv2.waitKey(100)
        '''
        return observation, self.automaton, self.image_locations, self.image_labels

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
            return self.image_locations[self._agent_location[0], self._agent_location[1]]

    def _get_obs(self, full = 1):
        return self._render_frame()

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
            blit_item(self.pickaxe_img, self._pickaxe_location, self._pickaxe_display)
            blit_item(self.gem_img, self._gem_location, self._gem_display)
            blit_item(self.door_img, self._exit_location)
            blit_item(self.lava_img, self._lava_location)
            blit_item(self.egg_img, self._egg_location)
            blit_item(self.robot_img, [self._agent_location], self._robot_display)

            #if self.render_mode == "human":
            #    cv2.imshow("Frame", canvas)
                # WaitKey delay is set based on desired render FPS.
                #key = cv2.waitKey(1)
                # Optionally, handle key input if needed.
            #else:
                # In "rgb_array" mode, return the canvas converted from BGR to RGB.
            return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()