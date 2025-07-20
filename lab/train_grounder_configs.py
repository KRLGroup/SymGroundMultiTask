from train_grounder import Args


train_grounder_base = Args(

    # Grounder parameters
    sym_grounder_model = "ObjectCNN",
    obs_size = (64,64),
    model_name = "sym_grounder_64_fixed",

    # Environment parameters
    env = "GridWorld-fixed-v1",
    test_env = "GridWorld-fixed-v1",
    ltl_sampler = "Dataset_e54",

    # Training parameters
    num_samples = 10000,
    batch_size = 32,
    seed = 1,

    # Agent parameters
    use_agent = False

)