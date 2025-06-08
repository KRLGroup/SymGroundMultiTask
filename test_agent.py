import pickle
import os
import argparse
import torch
import time

import utils
from ac_model import ACModel
from recurrent_ac_model import RecurrentACModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--agent_dir", default="storage/RGCN_8x32_ROOT_SHARED_None_GridWorld-v0_seed:1_epochs:4_bs:256_fpp:None_dsc:0.94_lr:0.0003_ent:0.01_clip:0.2_prog:full")
parser.add_argument("--formula_id", default=0, type=int)
args = parser.parse_args()

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(MAIN_DIR, args.agent_dir)


# load training config
with open(os.path.join(AGENT_DIR, "train/config.pickle"), "rb") as f:
    config = pickle.load(f)
print(f"\nConfig:\n{config}")

# build environment
env = utils.make_env(
    config.env,
    progression_mode=config.progression_mode,
    ltl_sampler=config.ltl_sampler,
    seed=1,
    intrinsic=config.int_reward,
    noLTL=config.noLTL,
    device=device
)
action_to_str = {0:"down", 1:"right", 2:"up", 3:"left"}

# set formula
env.env.produced_tasks = args.formula_id


# load training status
try:
    status = utils.get_status(AGENT_DIR + "/train", device)
except OSError:
    status = {"num_frames": 0, "update": 0}

# load observations preprocessor
using_gnn = (config.gnn != "GRU" and config.gnn != "LSTM")
obs_space, preprocess_obss = utils.get_obss_preprocessor(env, using_gnn, config.progression_mode)
if "vocab" in status and preprocess_obss.vocab is not None:
    preprocess_obss.vocab.load_vocab(status["vocab"])

# create model
use_mem = config.recurrence > 1
if use_mem:
    acmodel = RecurrentACModel(env.env, obs_space, env.action_space, config.ignoreLTL, config.gnn, config.dumb_ac, config.freeze_ltl)
else:
    acmodel = ACModel(env.env, obs_space, env.action_space, config.ignoreLTL, config.gnn, config.dumb_ac, config.freeze_ltl, device)

# load model
if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])
elif config.pretrained_gnn:
    acmodel.load_pretrained_gnn(pretrained_status["model_state"])

acmodel.to(device)

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

    preprocessed_obs = preprocess_obss([obs], device=device)
    dist, _ = acmodel(preprocessed_obs)
    a = torch.argmax(dist.logits, dim=-1).item()
    print(action_to_str[a])

    obs, reward, done, info = env.step(a)

    if done:
        env.show()
        print(f"Reward: {reward}")
        print("Done!")
        print("Closing...")
        break

    print(f"Reward: {reward}")

time.sleep(2)
env.close()