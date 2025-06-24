import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Tuple

import utils
from utils import EarlyStopping
from ReplayBuffer import ReplayBuffer
from DeepAutoma import MultiTaskProbabilisticAutoma
from envs.gridworld_multitask.Environment import GridWorldEnv_multitask


@dataclass
class Args:

    # Grounder parameters
    sym_grounder_model: str = "ObjectCNN"
    obs_size: Tuple[int,int] = (64,64)
    model_name: str = "sym_grounder"

    # Environment parameters
    max_num_steps: int = 50
    randomize_loc: bool = False
    randomize_test_loc: bool = False

    # Training parameters
    num_samples: int = 10000
    batch_size: int = 32
    seed: int = 1


REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def train_grounder(args: Args, device: str = None):

    device = torch.device(device) or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer = ReplayBuffer()

    # create model dir
    storage_dir = os.path.join(REPO_DIR, "storage")
    model_dir = os.path.join(storage_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)

    txt_logger = utils.get_txt_logger(model_dir)
    utils.save_config(model_dir, args)

    # log script arguments
    txt_logger.info("\n---\n")
    txt_logger.info("Args:")
    for field_name, value in vars(args).items():
        txt_logger.info(f"\t{field_name}: {value}")
    txt_logger.info(f"\nDevice: {device}")
    txt_logger.info("\n---\n")

    # set seed for all randomness sources
    utils.set_seed(args.seed)


    # INITIALIZATION

    txt_logger.info("Initialization\n")

    # environment used for training
    env = GridWorldEnv_multitask(
        state_type = "image",
        obs_size = args.obs_size,
        max_num_steps = args.max_num_steps,
        randomize_loc = args.randomize_loc
    )
    n_propositions = len(env.dictionary_symbols)
    txt_logger.info("-) Environment loaded.")

    # environent used for testing and logging about the symbol grounder
    test_env = GridWorldEnv_multitask(
        state_type = "image",
        obs_size = args.obs_size,
        randomize_loc = args.randomize_test_loc
    )
    txt_logger.info("-) Test environment loaded.")

    # create model
    sym_grounder = utils.make_grounder(args.sym_grounder_model, n_propositions, args.obs_size)
    sym_grounder.to(device)
    txt_logger.info("-) Grounder loaded.")

    # load previous training status
    status = utils.get_status(model_dir, device)
    txt_logger.info("-) Looking for status of previous training.")
    if status == None:
        status = {"epoch": 0}
        txt_logger.info("-) Previous status not found.")
    else:
        txt_logger.info("-) Previous status found.")

    # load existing model
    if "grounder_state" in status:
        sym_grounder.load_state_dict(status["grounder_state"])
        txt_logger.info("-) Loading grounder from existing run.")

    # setup optimizer (train the grounder and not the DeepDFA)
    optimizer = torch.optim.Adam(sym_grounder.parameters(), lr=0.001)
    cross_entr = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()

    # load optimizer of existing model
    if "optimizer_state" in status:
        optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("-) Loading optimizer from existing run.")

    txt_logger.info("-) Optimizer loaded.")
    txt_logger.info("\n---\n")


    # TRAINING

    txt_logger.info("Training\n")

    epoch = 0
    n_won = 0
    n_failed = 0
    n_episodes = 0

    loss_values = []
    test_class_accs = []
    train_class_accs = []

    # training loop
    while n_episodes < args.num_samples:
        n_episodes += 1

        # reset environments
        obs, task, train_env_images, train_env_labels = env.reset()
        _, _, test_env_images, test_env_labels = test_env.reset()

        # agent starts in an empty cell (never terminates in 0 actions)
        done = False
        episode_obss = [obs]
        episode_rews = [0]

        # play the episode until termination
        while not done:
            action = env.action_space.sample()
            obs, rw, done = env.step(action)
            episode_obss.append(obs)
            episode_rews.append(rw)

        # if the rewards are all 0 there is no supervision
        if rw != 0:

            if rw == 1:
                n_won += 1
            elif rw == -1:
                n_failed += 1

            txt_logger.info(f"won {n_won} tasks and failed {n_failed} over {n_episodes}")

            # extend shorter vectors to the max length
            if len(episode_rews) < env.max_num_steps+1:
                last_rew = episode_rews[-1]
                last_obs = episode_obss[-1]
                extension_length = env.max_num_steps+1 - len(episode_rews)
                episode_rews.extend([last_rew] * extension_length)
                episode_obss.extend([last_obs] * extension_length)

            # add to the buffer
            obss = np.stack(episode_obss)
            obss = torch.tensor(obss, device=device, dtype=torch.float64)
            dfa_trans = task.transitions
            dfa_rew = task.rewards
            rews = torch.LongTensor(episode_rews)
            buffer.push(obss, rews, dfa_trans, dfa_rew)

        # at each iteration train the sym_grounder after the buffer is full enough
        if len(buffer) >= 10 * args.batch_size:

            # sample from the buffer
            obss, rews, dfa_trans, dfa_rew = buffer.sample(args.batch_size)

            # build the differentiable reward machine for the task
            deepDFA = MultiTaskProbabilisticAutoma(
                batch_size = args.batch_size,
                numb_of_actions = n_propositions,
                numb_of_states = max([len(tr.keys()) for tr in dfa_trans]),
                reward_type = "ternary"
            )
            deepDFA.initFromDfas(dfa_trans, dfa_rew)

            txt_logger.info(f"\nEpoch {epoch}")
            epoch += 1

            # TRAINING STEP

            optimizer.zero_grad()

            # obtain probability of symbols from observations with sym_grounder
            symbols = sym_grounder(obss.view(-1, *obss.shape[2:]))
            symbols = symbols.view(*obss.shape[:2], -1)

            # predict state and reward from predicted symbols with DeepDFA
            pred_states, pred_rew = deepDFA(symbols)
            pred = pred_rew.view(-1, deepDFA.numb_of_rewards)

            # maps rewards to label
            labels = (rews + 1).view(-1)

            loss = cross_entr(pred, labels)

            # update sym_grounder
            loss.backward()
            optimizer.step()

            # LOGGING

            # collect data to compute accuracy on the train enviornment (only for logging)
            train_images = []
            train_labels = []
            for c in range(7):
                for r in range(7):
                    train_images.append(train_env_images[r, c])
                    train_labels.append(train_env_labels[r, c])
            train_images = np.stack(train_images)
            train_images = torch.tensor(train_images, device=device, dtype=torch.float64)
            train_labels = torch.LongTensor(train_labels).to(device)

            # collect data to compute accuracy on the test enviornment (only for logging)
            test_images = []
            test_labels = []
            for c in range(7):
                for r in range(7):
                    test_images.append(test_env_images[r, c])
                    test_labels.append(test_env_labels[r, c])
            test_images = np.stack(test_images)
            test_images = torch.tensor(test_images, device=device, dtype=torch.float64)
            test_labels = torch.LongTensor(test_labels).to(device)

            pred_sym_train = torch.argmax(sym_grounder(train_images), dim=-1)
            train_correct_preds = torch.sum((pred_sym_train == train_labels).long())
            train_class_acc = torch.mean((pred_sym_train == train_labels).float())

            pred_sym_test = torch.argmax(sym_grounder(test_images), dim=-1)
            test_correct_preds = torch.sum((pred_sym_test == test_labels).long())
            test_class_acc = torch.mean((pred_sym_test == test_labels).float())

            txt_logger.info(f"loss: {loss.item():.4e}")
            txt_logger.info(f"grounder TRAIN accuracy = {train_correct_preds.item()} / {pred_sym_train.shape[0]} ({train_class_acc.item():.4f})")
            txt_logger.info(f"grounder TEST accuracy = {test_correct_preds.item()} / {pred_sym_test.shape[0]} ({test_class_acc.item():.4f})")

            loss_values.append(loss.item())
            test_class_accs.append(test_class_acc.item())
            train_class_accs.append(train_class_acc.item())

            # every 10 epochs print comparison between true and predicted labels
            if epoch % 10 == 0:

                txt_logger.info("\n---")
                txt_logger.info("Comparison:")
                txt_logger.info("Train")
                txt_logger.info(f"true: {train_labels.tolist()}")
                txt_logger.info(f"pred: {pred_sym_train.tolist()}")
                txt_logger.info("Test")
                txt_logger.info(f"true: {test_labels.tolist()}")
                txt_logger.info(f"pred: {pred_sym_test.tolist()}")
                txt_logger.info("---")

            # every 100 epochs plot loss and accuracies and save the sym_grounder model
            if epoch % 100 == 0:

                plt.figure(figsize=(8, 5))
                plt.plot(loss_values, label='Loss', color='blue')
                plt.title("Training Loss Over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, f"loss.png"))
                plt.cla()
                plt.clf()
                plt.close()

                plt.figure(figsize=(8, 5))
                plt.plot(test_class_accs, label="Test Accuracy", color="red")
                plt.plot(train_class_accs, label="Train Accuracy", color="green")
                plt.title("Classification Accuracy Over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.ylim(-0.1, 1.1)
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, f"accuracy.png"))
                plt.cla()
                plt.clf()
                plt.close()

                status = {
                    "epoch": epoch,
                    "optimizer_state": optimizer.state_dict(),
                    "grounder_state": sym_grounder.state_dict()
                }

                utils.save_status(status, model_dir)
                txt_logger.info("Status saved")