from train_agent import Args


test_ltl2action = Args(
    algo = "ppo",
    env = "Letter-7x7-v3",
    log_interval = 5,
    save_interval = 20,
    frames = 20000000,
    discount = 0.94,
    ltl_sampler = "Eventually_1_5_1_4",
    epochs = 4,
    lr = 0.0003,
    procs = 1
)


# TODO num_envs should be 1 since we use shared pool of formulas?
test_gridworld = Args(

    # General parameters
    model_name = None,
    algo = "ppo",
    env = "GridWorld-v0",
    ltl_sampler = "None",
    dataset = "e54",
    log_interval = 10,
    save_interval = 100,
    procs = 1,
    frames = 20000000,
    gnn  =  "RGCN_8x32_ROOT_SHARED",

    # Evaluation parameters
    eval = True,
    eval_env = "GridWorld-v0",
    eval_interval = 500,
    ltl_samplers_eval = ['None', 'None'],
    eval_datasets = ["e54test", "e65test"],
    eval_episodes = [1000, 50],
    eval_procs = 1,

    # Parameters for main algorithm
    epochs = 4,
    discount = 0.94,
    lr = 0.0003,
    pretrained_gnn = True,
    pretrain_name = "pretrain"

)


# pretrain for GridWorld
test_simple_ltl_6l = Args(

    # General parameters
    algo = "ppo",
    env = "Simple-LTL-Env-6L-v0",
    ltl_sampler = "Eventually_1_5_1_4",
    log_interval = 5,
    save_interval = 20,
    procs = 1,
    frames = 20000000,
    gnn  =  "RGCN_8x32_ROOT_SHARED",

    # Parameters for main algorithm
    epochs = 4,
    discount = 0.94,
    lr = 0.0003,
    dumb_ac = True

)