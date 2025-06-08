import time
import datetime
import torch
import torch_ac
import tensorboardX
import sys
import glob
from math import floor
from dataclasses import dataclass, field
from typing import List, Optional

import utils
from ac_model import ACModel
from recurrent_ac_model import RecurrentACModel
from envs.gym_letters.letter_env import LetterEnv


# Parse arguments

@dataclass
class Args:
    # General parameters
    algo: str # a2c or ppo
    env: str
    ltl_sampler: str = "Default" # Must be None for GridWorld
    model: Optional[str] = None
    seed: int = 1
    log_interval: int = 10
    save_interval: int = 100
    procs: int = 16
    frames: int = 2 * 10**8
    checkpoint_dir: Optional[str] = None

    # Evaluation parameters
    eval: bool = False
    eval_episodes: int = 5
    eval_env: Optional[str] = None
    ltl_samplers_eval: Optional[List[str]] = None # at least 1 if present
    eval_procs: int = 1

    # Parameters for main algorithm
    epochs: int = 4
    batch_size: int = 256
    frames_per_proc: Optional[int] = None
    discount: float = 0.99
    lr: float = 0.0003
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    optim_eps: float = 1e-8
    optim_alpha: float = 0.99
    clip_eps: float = 0.2
    ignoreLTL: bool = False
    noLTL: bool = False
    progression_mode: str = "full" # full, partial, or none
    recurrence: int = 1
    gnn: str = "RGCN_8x32_ROOT_SHARED"
    int_reward: float = 0.0
    pretrained_gnn: bool = False
    dumb_ac: bool = False
    freeze_ltl: bool = False



def train_agent(args: Args, device: str = None):

    # SETUP

    use_mem = args.recurrence > 1
    device = torch.device(device) or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

    # Set run dir
    gnn_name = args.gnn
    if args.ignoreLTL:
        gnn_name = "IgnoreLTL"
    if args.dumb_ac:
        gnn_name = gnn_name + "-dumb_ac"
    if args.pretrained_gnn:
        gnn_name = gnn_name + "-pretrained"
    if args.freeze_ltl:
        gnn_name = gnn_name + "-freeze_ltl"
    if use_mem:
        gnn_name = gnn_name + "-recurrence:%d"%(args.recurrence)

    default_model_name = f"{gnn_name}_{args.ltl_sampler}_{args.env}_seed:{args.seed}_epochs:{args.epochs}_bs:{args.batch_size}_fpp:{args.frames_per_proc}_dsc:{args.discount}_lr:{args.lr}_ent:{args.entropy_coef}_clip:{args.clip_eps}_prog:{args.progression_mode}"

    model_name = args.model or default_model_name
    storage_dir = "storage" if args.checkpoint_dir is None else args.checkpoint_dir
    model_dir = utils.get_model_dir(model_name, storage_dir)

    pretrained_model_dir = None
    if args.pretrained_gnn:
        assert(args.progression_mode == "full")
        default_dir = f"symbol-storage/{args.gnn}-dumb_ac_{args.ltl_sampler}_Simple-LTL-Env-v0_seed:{args.seed}_*_prog:{args.progression_mode}/train"
        print(default_dir)
        model_dirs = glob.glob(default_dir)
        if len(model_dirs) == 0:
            raise Exception("Pretraining directory not found.")
        elif len(model_dirs) > 1:
            raise Exception("More than 1 candidate pretraining directory found.")
        pretrained_model_dir = model_dirs[0]

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir + "/train")
    csv_file, csv_logger = utils.get_csv_logger(model_dir + "/train")
    tb_writer = tensorboardX.SummaryWriter(model_dir + "/train")
    utils.save_config(model_dir + "/train", args)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Load environments
    envs = []
    progression_mode = args.progression_mode
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, progression_mode, args.ltl_sampler, args.seed, args.int_reward, args.noLTL, device))

    # Sync environments
    envs[0].reset()
    if isinstance(envs[0].env, LetterEnv):
        txt_logger.info("Using fixed maps.")
        for env in envs:
            env.env.map = envs[0].env.map

    txt_logger.info("Environments loaded\n")

    # Load training status
    try:
        status = utils.get_status(model_dir + "/train", device)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded.\n")

    if pretrained_model_dir is not None:
        try:
            pretrained_status = utils.get_status(pretrained_model_dir, device)
        except:
            txt_logger.info("Failed to load pretrained model.\n")
            exit(1)

    # Load observations preprocessor
    using_gnn = (args.gnn != "GRU" and args.gnn != "LSTM")
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0], using_gnn, progression_mode)
    if "vocab" in status and preprocess_obss.vocab is not None:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded.\n")

    # Load model
    if use_mem:
        acmodel = RecurrentACModel(envs[0].env, obs_space, envs[0].action_space, args.ignoreLTL, args.gnn, args.dumb_ac, args.freeze_ltl)
    else:
        acmodel = ACModel(envs[0].env, obs_space, envs[0].action_space, args.ignoreLTL, args.gnn, args.dumb_ac, args.freeze_ltl, device)

    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
        txt_logger.info("Loading model from existing run.\n")
    elif args.pretrained_gnn:
        acmodel.load_pretrained_gnn(pretrained_status["model_state"])
        txt_logger.info("Pretrained model loaded.\n")

    acmodel.to(device)
    txt_logger.info("Model loaded.\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo
    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Loading optimizer from existing run.\n")
    txt_logger.info("Optimizer loaded.\n")

    # init the evaluator
    if args.eval:
        eval_samplers = args.ltl_samplers_eval if args.ltl_samplers_eval else [args.ltl_sampler]
        eval_env = args.eval_env if args.eval_env else args.env
        eval_procs = args.eval_procs if args.eval_procs else args.procs

        evals = []
        for eval_sampler in eval_samplers:
            evals.append(utils.Eval(eval_env, model_name, eval_sampler,
                        seed=args.seed, device=device, num_procs=eval_procs, ignoreLTL=args.ignoreLTL, progression_mode=progression_mode, gnn=args.gnn, dumb_ac = args.dumb_ac))

    # TRAINING

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:

        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)

            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            average_reward_per_step = utils.average_reward_per_step(logs["return_per_episode"], logs["num_frames_per_episode"])
            average_discounted_return = utils.average_discounted_return(logs["return_per_episode"], logs["num_frames_per_episode"], args.discount)
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["average_reward_per_step", "average_discounted_return"]
            data += [average_reward_per_step, average_discounted_return]
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            # μ: mean | σ: std | m: min | M: max
            # U: update | F: frames | FPS | D: duration | rR: reshaped return | ARPS: average reward per step | ADR: average discounted return
            # F: num frames | H: entropy | V: value | pL: policy loss | vL: value loss | nabla: grad norm
            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | ARPS: {:.3f} | ADR: {:.3f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                    "model_state": algo.acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab") and preprocess_obss.vocab is not None:
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir + "/train")
            txt_logger.info("Status saved")

            if args.eval:
                # we send the num_frames to align the eval curves with the training curves on TB
                for evalu in evals:
                    evalu.eval(num_frames, episodes=args.eval_episodes)