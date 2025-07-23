import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
import pickle

import utils
from replay_buffer import ReplayBuffer
from deep_automa import MultiTaskProbabilisticAutoma


@dataclass
class Args:

    # Grounder parameters
    sym_grounder_model: str = "ObjectCNN"
    obs_size: Tuple[int,int] = (64,64)
    model_name: str = "sym_grounder"

    # Environment parameters
    max_num_steps: int = 50
    env: str = "GridWorld-fixed-v1"
    test_env: str = "GridWorld-fixed-v1"
    ltl_sampler: str = "Dataset_e54"

    # Training parameters
    num_samples: int = 10000
    batch_size: int = 32
    seed: int = 1

    # Agent parameters
    use_agent: bool = False
    agent_dir: Optional[str] = None
    agent_prob: float = 1.0


REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def train_grounder(args: Args, device: str = None):

    device = torch.device(device) or torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_env = utils.make_env(
        args.env,
        progression_mode = "full",
        ltl_sampler = args.ltl_sampler,
        grounder = None,
        obs_size = args.obs_size
    )
    train_env.env.max_num_steps = args.max_num_steps
    num_symbols = len(train_env.propositions)
    txt_logger.info("-) Environment loaded.")

    # environent used for testing and logging about the symbol grounder
    test_env = utils.make_env(
        args.test_env,
        progression_mode = "full",
        ltl_sampler = args.ltl_sampler,
        grounder = None,
        obs_size = args.obs_size
    )
    txt_logger.info("-) Test environment loaded.")

    # load agent
    agent = None
    if args.use_agent:

        agent_dir = os.path.join(storage_dir, args.agent_dir)
        with open(os.path.join(agent_dir, "config.pickle"), "rb") as f:
            config = pickle.load(f)

        agent = utils.Agent(
            train_env,
            train_env.observation_space,
            train_env.action_space,
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

    # create model
    sym_grounder = utils.make_grounder(args.sym_grounder_model, num_symbols, args.obs_size)
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

    # define loss and setup optimizer (train the grounder and not the DeepDFA)
    optimizer = torch.optim.Adam(sym_grounder.parameters(), lr=0.001)
    cross_entr = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()

    buffer = ReplayBuffer(device=device)

    # load optimizer of existing model
    if "optimizer_state" in status:
        optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("-) Loading optimizer from existing run.")

    txt_logger.info("-) Optimizer loaded.")
    txt_logger.info("\n---\n")


    # TRAINING

    txt_logger.info("Training\n")

    epoch = status["epoch"]
    n_won = 0
    n_failed = 0
    n_episodes = 0

    loss_values = []
    test_accs = []
    train_accs = []

    start_time = time.time()

    # training loop
    while n_episodes < args.num_samples:
        n_episodes += 1

        # reset environments
        obs = train_env.reset()
        test_env.reset()

        # choose whether to use agent or random
        agent_ep = (args.use_agent and np.random.rand() <= args.agent_prob)

        # agent starts in an empty cell (never terminates in 0 actions)
        done = False
        obss = [obs['features']]
        rews = [0]

        # play the episode until termination
        while not done:
            action = agent.get_action(obs).item() if agent_ep else train_env.action_space.sample()
            obs, rew, done, _ = train_env.step(action)
            obss.append(obs['features'])
            rews.append(rew)

        # if the rewards are all 0 there is no supervision
        if rew != 0:

            if rew == 1:
                n_won += 1
            elif rew == -1:
                n_failed += 1

            txt_logger.info(f"won {n_won} tasks and failed {n_failed} over {n_episodes}")

            # extend shorter vectors to the max length
            if len(rews) < train_env.max_num_steps+1:
                last_rew = rews[-1]
                last_obs = obss[-1]
                extension = train_env.max_num_steps+1 - len(rews)
                rews.extend([last_rew] * extension)
                obss.extend([last_obs] * extension)

            # add to the buffer
            obss = torch.tensor(np.stack(obss), device=device, dtype=torch.float32)
            rews = torch.tensor(rews, device=device, dtype=torch.int64)
            task = train_env.sampler.get_current_automaton()
            buffer.push(obss, rews, task.transitions, task.rewards)

        # after the buffer is full enough after each episode train the sym_grounder
        if len(buffer) >= 10 * args.batch_size:

            txt_logger.info(f"\nEpoch {epoch}")
            epoch += 1

            # TRAINING STEP

            # sample from the buffer
            obss, rews, dfa_trans, dfa_rew = buffer.sample(args.batch_size)

            # build the differentiable reward machine for the task
            deepDFA = MultiTaskProbabilisticAutoma(
                batch_size = args.batch_size,
                numb_of_actions = num_symbols,
                numb_of_states = max([len(tr.keys()) for tr in dfa_trans]),
                reward_type = "ternary",
                device = device
            )
            deepDFA.initFromDfas(dfa_trans, dfa_rew)

            # reset gradient
            optimizer.zero_grad()

            # obtain probability of symbols from observations with sym_grounder
            symbols = sym_grounder(obss.view(-1, *obss.shape[2:]))
            symbols = symbols.view(*obss.shape[:2], -1)

            # predict state and reward from predicted symbols with DeepDFA
            pred_states, pred_rew = deepDFA(symbols)
            pred = pred_rew.view(-1, deepDFA.numb_of_rewards)

            # maps rewards to label
            labels = (rews + 1).view(-1)

            # compute loss
            loss = cross_entr(pred, labels)

            # update sym_grounder
            loss.backward()
            optimizer.step()

            # LOGGING

            coords = train_env.env.loc_to_label.keys()

            # collect data to compute accuracy on the train enviornment (only for logging)
            train_images = np.stack([train_env.env.loc_to_obs[(r, c)] for (r, c) in coords])
            train_images = torch.tensor(train_images, device=device, dtype=torch.float32)
            train_true_syms = [train_env.env.loc_to_label[(r, c)] for (r, c) in coords]
            train_true_syms = torch.tensor(train_true_syms, device=device, dtype=torch.int32)

            # collect data to compute accuracy on the test enviornment (only for logging)
            test_images = np.stack([test_env.env.loc_to_obs[(r, c)] for (r, c) in coords])
            test_images = torch.tensor(test_images, device=device, dtype=torch.float32)
            test_true_syms = [test_env.env.loc_to_label[(r, c)] for (r, c) in coords]
            test_true_syms = torch.tensor(test_true_syms, device=device, dtype=torch.int32)

            # compute predictions on train environment
            train_pred_syms = torch.argmax(sym_grounder(train_images), dim=-1)
            train_correct_preds = torch.sum((train_pred_syms == train_true_syms).int())
            train_acc = torch.mean((train_pred_syms == train_true_syms).float())

            # compute predictions on test environment
            test_pred_syms = torch.argmax(sym_grounder(test_images), dim=-1)
            test_correct_preds = torch.sum((test_pred_syms == test_true_syms).int())
            test_acc = torch.mean((test_pred_syms == test_true_syms).float())

            duration = int(time.time() - start_time)

            txt_logger.info(f"loss: {loss.item():.4e} | duration: {duration:05}")
            txt_logger.info(f"grounder TRAIN accuracy = {train_correct_preds.item()} / {train_pred_syms.shape[0]} ({train_acc.item():.4f})")
            txt_logger.info(f"grounder TEST accuracy = {test_correct_preds.item()} / {test_pred_syms.shape[0]} ({test_acc.item():.4f})")

            loss_values.append(loss.item())
            test_accs.append(test_acc.item())
            train_accs.append(train_acc.item())

            # every 10 epochs print comparison between true and predicted labels
            if epoch % 10 == 0:

                txt_logger.info("\n---")
                txt_logger.info("Comparison:")
                txt_logger.info("Train")
                txt_logger.info(f"true: {train_true_syms.tolist()}")
                txt_logger.info(f"pred: {train_pred_syms.tolist()}")
                txt_logger.info("Test")
                txt_logger.info(f"true: {test_true_syms.tolist()}")
                txt_logger.info(f"pred: {test_pred_syms.tolist()}")
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
                plt.plot(test_accs, label="Test Accuracy", color="red")
                plt.plot(train_accs, label="Train Accuracy", color="green")
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