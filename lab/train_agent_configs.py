from train_agent import Args


test_gridworld = Args(

    # General parameters
    model_name = "agent",
    algo = "ppo",
    seed = 1,
    log_interval = 10,
    save_interval = 100,
    procs = 1,
    frames = 20000000,

    # Environment parameters
    env = "GridWorld-fixed-v1",
    state_type = 'image',
    obs_size = (56,56),
    ltl_sampler = "Dataset_e54",

    # GNN parameters
    gnn_model = "RGCN_8x32_ROOT_SHARED",
    use_pretrained_gnn = True,
    gnn_pretrain = "gnn_pretrain",
    freeze_gnn = False,

    # Grounder parameters
    grounder_model = "ObjectCNN",
    use_pretrained_grounder = True,
    grounder_pretrain = "full_grounder_56",
    freeze_grounder = True,

    # Agent parameters

    # Evaluation parameters
    eval = True,
    eval_env = "GridWorld-fixed-v1",
    eval_interval = 1000,
    eval_samplers = ['Dataset_e54test', 'Dataset_e65test'],
    eval_episodes = [1000, 50],
    eval_procs = 1,

    # Train parameters
    epochs = 4,
    batch_size = 256,
    discount = 0.94,
    lr = 3e-4,
    gae_lambda = 0.95,
    entropy_coef = 0.01,
    value_loss_coef = 0.5,
    max_grad_norm = 0.5,
    clip_eps = 0.2,

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
    epochs = 2,
    batch_size = 1024,
    discount = 0.9,
    lr = 1e-3,
    gae_lambda = 0.5,
    entropy_coef = 0.01,
    value_loss_coef = 0.5,
    max_grad_norm = 0.5,
    clip_eps = 0.1

)