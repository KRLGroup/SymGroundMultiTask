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


test_gridworld = Args(

    # General parameters
    model_name = None,
    algo = "ppo",
    seed = 1,
    log_interval = 10,
    save_interval = 100,
    procs = 1,
    frames = 20000000,

    # Environment parameters
    env = "GridWorld-fixed-v1",
    obs_size = (56,56),
    ltl_sampler = "Dataset_e54",

    # GNN parameters
    gnn_model = "RGCN_8x32_ROOT_SHARED",
    use_pretrained_gnn = True,
    gnn_pretrain = "pretrain",
    freeze_gnn = False,

    # Grounder parameters
    grounder_model = "ObjectCNN",
    use_pretrained_grounder = True,
    grounder_pretrain = "sym_grounder_56_fixed",
    freeze_grounder = True,

    # Agent parameters

    # Evaluation parameters
    eval = True,
    eval_env = "GridWorld-fixed-v1",
    eval_interval = 500,
    eval_samplers = ['Dataset_e54test', 'Dataset_e65test'],
    eval_episodes = [1000, 50],
    eval_procs = 1,

    # Train parameters
    epochs = 4,
    discount = 0.94,
    lr = 0.0003,

)


# pretrain for GridWorld
test_simple_ltl_5l = Args(

    # General parameters
    model_name = "gnn_pretrain",
    algo = "ppo",
    seed = 1,
    log_interval = 10,
    save_interval = 100,
    procs = 1,
    frames = 20000000,

    # Environment parameters
    env = "Simple-LTL-Env-5L-v0",
    obs_size = (56,56),
    ltl_sampler = "Eventually_1_5_1_4",

    # GNN parameters
    gnn_model = "RGCN_8x32_ROOT_SHARED",
    use_pretrained_gnn = False,
    gnn_pretrain = None,
    freeze_gnn = False,

    # Grounder parameters
    grounder_model = None,

    # Agent parameters
    dumb_ac = True,

    # Evaluation parameters
    eval = True,
    eval_env = "Simple-LTL-Env-5L-v0",
    eval_interval = 100,
    eval_samplers = ['Eventually_1_5_1_4'],
    eval_episodes = [500],
    eval_procs = 1,

    # Train parameters
    epochs = 4,
    discount = 0.94,
    lr = 0.0003,

)