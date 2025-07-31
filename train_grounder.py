import torch
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import pickle
from tqdm import tqdm
import tensorboardX

import utils
from grounder_algo import GrounderAlgo


@dataclass
class Args:

    # General parameters
    model_name: str = "sym_grounder"
    log_interval: int = 1
    save_interval: int = 10
    seed: int = 1

    # Grounder parameters
    sym_grounder_model: str = "ObjectCNN"
    obs_size: Tuple[int,int] = (56,56)

    # Environment parameters
    max_num_steps: int = 50
    env: str = "GridWorld-fixed-v1"
    ltl_sampler: str = "Dataset_e54"

    # Training parameters
    updates: int = 10000
    buffer_size: int = 1000
    buffer_start: int = 32
    max_env_steps: int = 75
    batch_size: int = 32
    lr: float = 0.001
    update_steps: int = 1

    # Evaluation parameters
    evaluate_steps: int = 1

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
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)
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
    sym_grounder = utils.make_grounder(
        model_name = args.sym_grounder_model,
        num_symbols = num_symbols,
        obs_size = args.obs_size,
        freeze_grounder = False
    )
    sym_grounder.to(device)
    txt_logger.info("-) Grounder loaded.")

    # load previous training status
    status = utils.get_status(model_dir, device)
    txt_logger.info("-) Looking for status of previous training.")
    if status == None:
        status = {"update": 0, "num_frames": 0}
        txt_logger.info("-) Previous status not found.")
    else:
        txt_logger.info("-) Previous status found.")

    # load existing model
    if "grounder_state" in status:
        sym_grounder.load_state_dict(status["grounder_state"])
        txt_logger.info("-) Loading grounder from existing run.")

    # load grounder algo
    grounder_algo = GrounderAlgo(sym_grounder, train_env, True, args.max_env_steps, args.buffer_size,
                                 args.batch_size, args.lr, args.update_steps, args.evaluate_steps, device)

    # load grounder optimizer of existing model
    if "grounder_optimizer_state" in status:
        grounder_algo.optimizer.load_state_dict(status["grounder_optimizer_state"])
        txt_logger.info("-) Loading grounder optimizer from existing run.")

    txt_logger.info("-) Grounder training algorithm loaded.")
    txt_logger.info("\n---\n")


    # TRAINING

    txt_logger.info("Training\n")

    update = status["update"]
    num_frames = status["num_frames"]
    start_time = time.time()

    # populate buffer
    txt_logger.info("Initializing Buffer...\n")
    progress = tqdm(total=args.buffer_start)
    while progress.n < args.buffer_start:
        logs = grounder_algo.collect_experiences()
        progress.n = logs['buffer']
        num_frames += logs['num_frames']
        progress.refresh()
    progress.close()

    # training loop
    while update < args.updates:

        # choose whether to use agent or random
        agent_ep = (args.use_agent and np.random.rand() <= args.agent_prob)
        train_env.env.sym_grounder = sym_grounder if agent_ep else None

        logs1 = grounder_algo.collect_experiences(agent = agent if agent_ep else None)
        logs2 = grounder_algo.update_parameters()

        update += 1
        num_frames += logs1["num_frames"]

        # logging

        if update % args.log_interval == 0:

            logs3 = grounder_algo.evaluate()
            logs = {**logs1, **logs2, **logs3}

            duration = int(time.time() - start_time)

            header = ["time/update", "time/frames", "time/duration"]
            data = [update, num_frames, duration]
            header += ["grounder/buffer", "grounder/loss", "grounder/acc"]
            data += [logs["buffer"], logs["grounder_loss"], logs["grounder_acc"]]
            header += [f"grounder_recall/{i}" for i in range(num_symbols)]
            data += logs["grounder_recall"]

            # U: update | F: frames | D: duration | B: buffer | L: loss | A: accuracy | R: recall
            format_str = (
                "U {:5} | F {:7} | D {:5} | B {:5} | L {:.6f} | A {:.3f}" +
                " | R" + "".join([" {:.2f}" for i in range(num_symbols)])
            )
            txt_logger.info(format_str.format(*data))

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # save status

        if update % args.save_interval == 0:

            status = {
                "update": update,
                "num_frames": num_frames,
                "grounder_optimizer_state": grounder_algo.optimizer.state_dict(),
                "grounder_state": sym_grounder.state_dict()
            }

            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")