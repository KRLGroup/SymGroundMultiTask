from train_agent import Args


baseline_gridworld = Args(

    # General parameters
    model_name = "baseline_compositional",
    algo = "compositional_ppo",
    seed = 1,
    log_interval = 1,
    save_interval = 10,
    procs = 16,
    frames_per_proc = 30,
    frames = 20000000,
    checkpoint_dir = None,

    # Environment parameters
    env = "GridWorld-v1",
    max_num_steps = 15,
    state_type = 'image',
    obs_size = (56,56),
    ltl_sampler = "Eventually_1_2_1_1",
    noLTL = False,
    progression_mode = "full",
    int_reward = 0.0,

    # GNN parameters
    ignoreLTL = False,
    gnn_model = None,
    use_pretrained_gnn = False,
    gnn_pretrain = None,
    freeze_gnn = False,

    # Grounder parameters
    grounder_model = None,
    use_pretrained_grounder = False,
    grounder_pretrain = None,
    freeze_grounder = False,

    # Agent parameters
    dumb_ac = False,
    recurrence = 1,
    compositional = True,

    # Evaluation parameters
    eval = False,
    eval_env = "GridWorld-v1",
    eval_interval = 500,
    eval_samplers = [],
    eval_episodes = [],
    eval_argmaxs = [],
    eval_procs = 1,

    # Train parameters
    epochs = 4,
    batch_size = 256,
    discount = 0.94,
    lr = 1e-4,
    gae_lambda = 0.95,
    entropy_coef = 0.1,
    value_loss_coef = 0.5,
    max_grad_norm = 0.5,
    optim_eps = 1e-8,
    optim_alpha = 0.99,
    clip_eps = 0.2,

    # Grounder training parameters
    grounder_buffer_size = 2048,
    grounder_buffer_start = 0,
    grounder_max_env_steps = 75,
    grounder_train_interval = 1,
    grounder_lr = 0.001,
    grounder_batch_size = 16,
    grounder_update_steps = 64,
    grounder_accumulation = 4,
    grounder_evaluate_steps = 256,
    grounder_use_early_stopping = True,
    grounder_patience = 250,
    grounder_min_delta = 0.0,

)