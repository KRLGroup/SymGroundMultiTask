import numpy as np
import enum
import gym
import torch


try:
    from safety_gym.envs.engine import Engine

except (ImportError, ModuleNotFoundError) as e:
    Engine = None
    _safety_gym_import_error = e


class zone(enum.Enum):
    JetBlack = 0
    White = 1
    Blue = 2
    Green = 3
    Red = 4
    Yellow = 5
    Cyan = 6
    Magenta = 7

    def __lt__(self, sth):
        return self.value < sth.value

    def __str__(self):
        return self.name[0].lower()

    def __repr__(self):
        return self.name


GROUP_ZONE = 7



if Engine is None:

    class ZonesEnv:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ZonesEnv requires MuJoCo and safety_gym to be installed.\n"
                f"Original error: {_safety_gym_import_error}"
            )


    class ZonesEnv_LTL2Action(ZonesEnv): pass
    class ZonesEnv1(ZonesEnv): pass
    class ZonesEnv1Fixed(ZonesEnv): pass
    class ZonesEnv5(ZonesEnv): pass
    class ZonesEnv5Fixed(ZonesEnv): pass



else:

    class ZonesEnv(Engine):
        """
        This environment is a modification of the Safety-Gym's environment.
        There is no "goal circle" but rather a collection of zones that the
        agent has to visit or to avoid in order to finish the task.

        For now we only support the 'point' robot.
        """

        metadata = {
            "state_types": ["image", "classic"],
        }

        all_zones_rgbs = {
                zone.JetBlack: [0, 0, 0, 1],
                zone.Blue : [0, 0, 1, 1],
                zone.Green : [0, 1, 0, 1],
                zone.Cyan : [0, 1, 1, 1],
                zone.Red : [1, 0, 0, 1],
                zone.Magenta : [1, 0, 1, 1],
                zone.Yellow : [1, 1, 0, 1],
                zone.White : [1, 1, 1, 1]
        }

        def __init__(self, zones, use_fixed_map, max_num_steps, state_type="classic", obs_size=(56,56), grounder=None):

            assert state_type in self.metadata["state_types"]
            self.obs_size = obs_size
            self.state_type = state_type
            self.max_num_steps = max_num_steps

            walled = True
            self.DEFAULT.update({
                'observe_zones': False,
                'zones_num': 0,  # Number of hazards in an environment
                'zones_placements': None,  # Placements list for hazards (defaults to full extents)
                'zones_locations': [],  # Fixed locations to override placements
                'zones_keepout': 0.55,  # Radius of hazard keepout for placement
                'zones_size': 0.25,  # Radius of hazards
            })

            if (walled):
                world_extent = 2.5
                walls = [(i/10, j) for i in range(int(-world_extent * 10),int(world_extent * 10 + 1),1) for j in [-world_extent, world_extent]]
                walls += [(i, j/10) for i in [-world_extent, world_extent] for j in range(int(-world_extent * 10), int(world_extent * 10 + 1),1)]
                self.DEFAULT.update({
                    'placements_extents': [-world_extent, -world_extent, world_extent, world_extent],
                    'walls_num': len(walls),  # Number of walls
                    'walls_locations': walls,  # This should be used and length == walls_num
                    'walls_size': 0.1,  # Should be fixed at fundamental size of the world
                })

            self.zones = zones
            self.zone_types = list(set(zones))
            self.zone_types.sort()
            self.dictionary_symbols = [str(i) for i in self.zone_types] + ['']

            self.use_fixed_map = use_fixed_map
            self.zone_rgbs = np.array([self.all_zones_rgbs[haz] for haz in self.zones])

            parent_config = {
                'robot_base': 'xmls/point.xml',
                'task': 'none',
                'lidar_num_bins': 16,
                'observe_zones': True,
                'zones_num': len(zones),
                'num_steps': max_num_steps
            }

            super().__init__(parent_config)


        @property
        def zones_pos(self):
            ''' Helper to get the zones positions from layout '''
            return [self.data.get_body_xpos(f'zone{i}').copy() for i in range(self.zones_num)]


        def build_observation_space(self):
            super().build_observation_space()

            if self.state_type == "image":
                self.observation_flatten = False
                self.obs_space_dict = {}
                self.observation_space = gym.spaces.Box(
                    np.float32(0.0),
                    np.float32(1.0),
                    (3,) + self.obs_size,
                    dtype=np.float32
                )

            else:

                if self.observe_zones:
                    for zone_type in self.zone_types:
                        self.obs_space_dict.update({f'zones_lidar_{zone_type}': gym.spaces.Box(
                            np.float32(0.0),
                            np.float32(1.0),
                            (self.lidar_num_bins,),
                            dtype=np.float32
                        )})

                if self.observation_flatten:
                    self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
                    self.observation_space = gym.spaces.Box(
                        np.float32(-np.inf),
                        np.float32(np.inf),
                        (self.obs_flat_size,),
                        dtype=np.float32
                    )

                else:
                    self.observation_space = gym.spaces.Dict(self.obs_space_dict)


        def build_placements_dict(self):
            super().build_placements_dict()
            if self.zones_num: #self.constrain_hazards:
                self.placements.update(self.placements_dict_from_object('zone'))


        def build_world_config(self):
            world_config = super().build_world_config()

            for i in range(self.zones_num):
                name = f'zone{i}'
                geom = {'name': name,
                        'size': [self.zones_size, 1e-2],#self.zones_size / 2],
                        'pos': np.r_[self.layout[name], 2e-2],#self.zones_size / 2 + 1e-2],
                        'rot': self.random_rot(),
                        'type': 'cylinder',
                        'contype': 0,
                        'conaffinity': 0,
                        'group': GROUP_ZONE,
                        'rgba': self.zone_rgbs[i] * [1, 1, 1, 0.25]} #0.1]}  # transparent
                world_config['geoms'][name] = geom

            return world_config


        def build_image_obs(self):
            vision = self.render(mode='rgb_array', width=self.obs_size[0], height=self.obs_size[1], camera_id=1)
            vision = np.array(vision, dtype='float32') / 255
            vision = vision.transpose(2, 0, 1)
            return vision


        def build_lidar_obs(self):
            obs = super().build_obs()
            if self.observe_zones:
                for zone_type in self.zone_types:
                    ind = [i for i, z in enumerate(self.zones) if (self.zones[i] == zone_type)]
                    pos_in_type = list(np.array(self.zones_pos)[ind])
                    obs[f'zones_lidar_{zone_type}'] = self.obs_lidar(pos_in_type, GROUP_ZONE)
            return obs


        def build_obs(self):
            obs = self.build_lidar_obs()
            if self.state_type == 'image':
                return self.build_image_obs()
            else:
                return obs


        def step(self, action):
            obs, reward, done, info = super().step(action)
            obs = obs.astype(np.float32)
            return obs, reward, done, info


        def render_lidars(self):
            offset = super().render_lidars()
            if self.render_lidar_markers:
                for zone_type in self.zone_types:
                    if f'zones_lidar_{zone_type}' in self.obs_space_dict:
                        ind = [i for i, z in enumerate(self.zones) if (self.zones[i] == zone_type)]
                        pos_in_type = list(np.array(self.zones_pos)[ind])
                        self.render_lidar(pos_in_type, np.array([self._rgb[zone_type]]), offset, GROUP_ZONE)
                        offset += self.render_lidar_offset_delta
            return offset


        def seed(self, seed=None):
            if (self.use_fixed_map): self._seed = seed



    # interface needed to build the ltl_wrapper
    class ZonesEnv_LTL2Action(ZonesEnv):

        def __init__(self, grounder, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.curr_step = 0
            self.num_episodes = 0
            self.sym_grounder = grounder
            self.current_obs = None


        def reset(self):
            self.num_episodes += 1
            self.curr_step = 0
            obs = super().reset()
            self.current_obs = obs
            return obs


        def step(self, action):
            self.curr_step += 1
            obs, rew, done, info = super().step(action)
            self.current_obs = obs
            return obs, rew, done, info


        def get_propositions(self):
            return self.dictionary_symbols[:-1].copy()


        def get_real_events(self):
            events = ""
            for h_inedx, h_pos in enumerate(self.zones_pos):
                h_dist = self.dist_xy(h_pos)
                if h_dist <= self.zones_size:
                    # We assume the agent to be in one zone at a time
                    events += str(self.zones[h_inedx])
            return events


        def get_events(self):

            # returns the proposition that currently holds
            if self.sym_grounder == None:
                return self.get_real_events()

            # returns the proposition that currently holds according to the grounder
            else:
                with torch.no_grad():
                    current_obs = torch.tensor(self.current_obs, device=self.sym_grounder.device).unsqueeze(0)
                    pred_sym = torch.argmax(self.sym_grounder(current_obs), dim=-1)[0]
                return self.dictionary_symbols[pred_sym]



    # Preconstructed Environments

    class ZonesEnv1(ZonesEnv_LTL2Action):
        def __init__(self, state_type, grounder, obs_size):
            super().__init__(
                state_type = state_type,
                grounder = grounder,
                obs_size = obs_size,
                zones = [zone.Red],
                use_fixed_map = False,
                max_num_steps = 1000
            )


    class ZonesEnv1Fixed(ZonesEnv_LTL2Action):
        def __init__(self, state_type, grounder, obs_size):
            super().__init__(
                state_type = state_type,
                grounder = grounder,
                obs_size = obs_size,
                zones = [zone.Red],
                use_fixed_map = True,
                max_num_steps = 1000
            )


    class ZonesEnv5(ZonesEnv_LTL2Action):
        def __init__(self, state_type, grounder, obs_size):
            super().__init__(
                state_type = state_type,
                grounder = grounder,
                obs_size = obs_size,
                zones = [zone.JetBlack, zone.JetBlack, zone.Red, zone.Red, zone.White, zone.White,  zone.Yellow, zone.Yellow],
                use_fixed_map = False,
                max_num_steps = 1000
            )


    class ZonesEnv5Fixed(ZonesEnv_LTL2Action):
        def __init__(self, state_type, grounder, obs_size):
            super().__init__(
                state_type = state_type,
                grounder = grounder,
                obs_size = obs_size,
                zones = [zone.JetBlack, zone.JetBlack, zone.Red, zone.Red, zone.White, zone.White,  zone.Yellow, zone.Yellow],
                use_fixed_map = True,
                max_num_steps = 1000
            )