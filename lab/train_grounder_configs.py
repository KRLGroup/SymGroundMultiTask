from train_grounder import Args


train_grounder_base = Args(

    # General parameters
    model_name = "grounder_with_algo",
    log_interval = 1,
    save_interval = 10,
    seed = 1,

    # Grounder parameters
    sym_grounder_model = "ObjectCNN",
    obs_size = (56,56),

    # Environment parameters
    max_num_steps = 50,
    env = "GridWorld-fixed-v1",
    eval_env = "GridWorld-fixed-v1",
    ltl_sampler = "Dataset_e54",

    # Training parameters
    updates = 10000,
    buffer_size = 1000,
    buffer_start = 32,
    max_env_steps = 75,
    batch_size = 32,
    lr = 0.001,
    update_steps = 4,

    # Agent parameters
    use_agent = False,
    agent_dir = None,
    agent_prob = 0.1

)