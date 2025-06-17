import argparse
import torch
import numpy as np

import utils
from envs.gridworld_multitask.Environment import GridWorldEnv_multitask


parser = argparse.ArgumentParser()
parser.add_argument('--agent_centric', default=True, dest='agent_centric', action='store_true')
parser.add_argument('--no-agent_centric', dest='agent_centric', action='store_false')
parser.add_argument("--iters", default=2000, type=int)
parser.add_argument("--grounder", default=None, type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# build environment
env = GridWorldEnv_multitask(state_type="image", max_num_steps=50, randomize_loc=True, agent_centric_view=args.agent_centric)

# load grounder
sym_grounder = torch.load(args.grounder, map_location=device)

# TEST

class_accs = []

for i in range(args.iters):

    print(f"iteration {i}")

    _, _, env_images, env_labels = env.reset()
    env.sampler.sampled_tasks = -1

    # collect data to compute accuracy on the test enviornment (only for logging)
    images = []
    labels = []
    for c in range(7):
        for r in range(7):
            images.append(env_images[r, c])
            labels.append(env_labels[r, c])
    images = np.stack(images)
    images = torch.tensor(images, device=device, dtype=torch.float64)
    # images = torch.stack(images, dim=0).to(device)
    labels = torch.LongTensor(labels).to(device)

    pred_sym = torch.argmax(sym_grounder(images), dim=-1)
    correct_preds = torch.sum((pred_sym == labels).long())
    class_acc = torch.mean((pred_sym == labels).float())

    print(f"grounder accuracy = {correct_preds.item()} / {pred_sym.shape[0]} ({class_acc.item():.4f})")

    class_accs.append(class_acc.item())
    mean_class_acc = torch.mean(torch.tensor(class_accs, device=device, dtype=torch.float64))
    print(f"cumulative accuracy = {mean_class_acc:.10f}")

    print("---")