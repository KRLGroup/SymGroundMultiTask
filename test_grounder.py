import argparse
import torch
import numpy as np
import os
import pickle

import utils


parser = argparse.ArgumentParser()
parser.add_argument("--device", default=None, type=str)
parser.add_argument('--env', default="GridWorld-fixed-v1", type=str)
parser.add_argument("--grounder", default=None, type=str)
parser.add_argument("--iters", default=2000, type=int)
args = parser.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(device)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


# compute grounder dir
storage_dir = os.path.join(REPO_DIR, "storage")
grounder_dir = os.path.join(storage_dir, args.grounder)

# load training config
with open(os.path.join(grounder_dir, "config.pickle"), "rb") as f:
    config = pickle.load(f)
print(f"\nConfig:\n{config}")

# build environment
env = utils.make_env(
    args.env,
    progression_mode = "full",
    ltl_sampler = "Dataset_e54",
    grounder = None,
    obs_size = config.obs_size
)

num_symbols = len(env.env.dictionary_symbols)

# load training status
status = utils.get_status(grounder_dir, device)

# load grounder
sym_grounder = utils.make_grounder(
    model_name = "ObjectCNN",
    num_symbols = num_symbols,
    obs_size = config.obs_size,
    freeze_grounder = True
)
sym_grounder.load_state_dict(status["grounder_state"])
sym_grounder.to(device)


# TEST

accs = []

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

    # predict symbols
    pred_syms = torch.argmax(sym_grounder(images), dim=-1)
    correct_preds = torch.sum((pred_syms == real_syms).int())
    acc = torch.mean((pred_syms == real_syms).float())

    print(f"grounder accuracy = {correct_preds.item()} / {pred_syms.shape[0]} ({acc.item():.4f})")

    accs.append(acc.item())
    mean_acc = torch.mean(torch.tensor(accs, device=device, dtype=torch.float32))
    print(f"cumulative accuracy = {mean_acc:.10f}")

    print("---")