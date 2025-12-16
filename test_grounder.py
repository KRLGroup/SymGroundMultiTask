import argparse
import torch
import numpy as np
import os

import utils


parser = argparse.ArgumentParser()
parser.add_argument("--device", default=None, type=str)
parser.add_argument('--env', default="GridWorld-v1", type=str)
parser.add_argument("--model_dir", default=None, type=str)
parser.add_argument("--iters", default=2000, type=int)
args = parser.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(device)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


# compute grounder dir
storage_dir = os.path.join(REPO_DIR, "storage")
grounder_dir = os.path.join(storage_dir, args.model_dir)

# load training config
config = utils.load_config(grounder_dir)
print(f"\nConfig:\n{config}")

# build environment
env = utils.make_env(args.env, config.progression_mode, config.ltl_sampler, config.seed, 0, False, config.state_type,
                     None, config.obs_size, config.max_num_steps)

obs_shape = env.observation_space['features'].shape
num_grounder_classes = len(env.propositions) + 1

num_symbols = len(env.env.dictionary_symbols)

# load training status
status = utils.get_status(grounder_dir, device)

# load grounder
sym_grounder = utils.make_grounder(config.grounder_model, num_grounder_classes, obs_shape, True)
sym_grounder.load_state_dict(status["grounder_state"])
sym_grounder.to(device)


# TEST

all_correct = 0
all_checked = 0

for i in range(args.iters):

    print(f"iteration {i}")

    env.reset()
    env.sampler.sampled_tasks = -1

    coords = env.env.loc_to_label.keys()

    # obtain and preprocess data
    images = np.stack([env.env.loc_to_obs[(r, c)] for (r, c) in coords])
    images = torch.tensor(images, device=device, dtype=torch.float32)
    real_syms = [env.env.loc_to_label[(r, c)] for (r, c) in coords]
    real_syms = torch.tensor(real_syms, device=device, dtype=torch.int32)

    # iteration accuracy
    pred_syms = torch.argmax(sym_grounder(images), dim=-1)
    correct_preds = torch.sum((pred_syms == real_syms)).item()
    acc = correct_preds / pred_syms.shape[0]

    print(f"accuracy = {correct_preds} / {pred_syms.shape[0]} ({acc:.4f})")

    # comulative accuracy
    all_correct += correct_preds
    all_checked += pred_syms.shape[0]
    mean_acc = all_correct / all_checked

    print(f"cumulative accuracy = {all_correct} / {all_checked} ({mean_acc:.10f})")

    print("---")