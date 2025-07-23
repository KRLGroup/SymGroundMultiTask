from train_grounder import Args


train_grounder_base = Args(

    # Grounder parameters
    sym_grounder_model = "ObjectCNN",
    obs_size = (56,56),
    model_name = "sym_grounder_56_fixed",

    # Environment parameters
    max_num_steps = 50,
    env = "GridWorld-fixed-v1",
    test_env = "GridWorld-fixed-v1",
    ltl_sampler = "Dataset_e54",

    # Training parameters
    num_samples = 10000,
    batch_size = 32,
    seed = 1,

    # Agent parameters
    use_agent = False,
    agent_dir = None,
    agent_prob = 1.0

)