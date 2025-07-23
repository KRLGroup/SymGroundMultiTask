import pickle
import os
import argparse
import torch
import time

import utils
from ac_model import ACModel
from recurrent_ac_model import RecurrentACModel


parser = argparse.ArgumentParser()
parser.add_argument("--device", default=None, type=str)
parser.add_argument("--agent_dir", default="RGCN_8x32_ROOT_SHARED-pretrained_Dataset_e54_GridWorld-v1_seed:1_epochs:4_bs:256_fpp:None_dsc:0.94_lr:0.0003_ent:0.01_clip:0.2_prog:full")
parser.add_argument("--ltl_sampler", default="Dataset_e54test_no-shuffle")
parser.add_argument("--formula_id", default=0, type=int)
args = parser.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(device)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


# compute agent dir
storage_dir = os.path.join(REPO_DIR, "storage")
agent_dir = os.path.join(storage_dir, args.agent_dir)

# load training config
with open(os.path.join(agent_dir, "config.pickle"), "rb") as f:
    config = pickle.load(f)
print(f"\nConfig:\n{config}")

# load training status
status = utils.get_status(agent_dir, device)

# build environment
env = utils.make_env(
    config.env,
    progression_mode = config.progression_mode,
    ltl_sampler = args.ltl_sampler,
    seed = 1,
    intrinsic = config.int_reward,
    noLTL = config.noLTL,
    grounder = None,
    obs_size = config.obs_size
)
action_to_str = {0:"down", 1:"right", 2:"up", 3:"left"}

# set formula
env.sampler.sampled_tasks = args.formula_id

# create and load grounder
sym_grounder = utils.make_grounder(config.grounder_model, len(env.propositions), config.obs_size)
sym_grounder.load_state_dict(status["grounder_state"]) if sym_grounder is not None else None
sym_grounder.to(device) if sym_grounder is not None else None
env.env.sym_grounder = sym_grounder

agent = utils.Agent(
    env,
    env.observation_space,
    env.action_space,
    agent_dir,
    config.ignoreLTL,
    config.progression_mode,
    config.gnn_model,
    recurrence = config.recurrence,
    dumb_ac = config.dumb_ac,
    device = device,
    argmax = True,
    num_envs = 1,
    verbose = False
)


# TEST

obs = env.reset()
done = False
step = 0

while not done:

    step += 1
    env.show()

    time.sleep(0.5)

    print(f"\n---")
    print(f"Step: {step}")
    print(f"Task:")
    utils.pprint_ltl_formula(env.translate_formula(obs['text']))

    print("\nAction: ", end="")

    action = agent.get_action(obs).item()
    print(action_to_str[action])

    obs, reward, done, info = env.step(action)

    if done:
        env.show()
        print(f"Reward: {reward}")
        print("Done!")
        print("Closing...")
        break

    print(f"Reward: {reward}")

time.sleep(2)
env.close()