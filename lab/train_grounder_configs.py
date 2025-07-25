from train_grounder import Args


train_grounder_base = Args(

    # General parameters
    model_name = "grounder_with_algo",
    log_interval = 1,
    save_interval = 10,

    # Grounder parameters
    sym_grounder_model = "ObjectCNN",
    obs_size = (56,56),

    # Environment parameters
    max_num_steps = 50,
    env = "GridWorld-fixed-v1",
    eval_env = "GridWorld-fixed-v1",
    ltl_sampler = "Dataset_e54",

    # Training parameters
    epochs = 10000,
    batch_size = 32,
    seed = 1,

    # Agent parameters
    use_agent = False,
    agent_dir = None,
    agent_prob = 0.1

)