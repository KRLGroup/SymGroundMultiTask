from train_grounder import Args


train_grounder_base = Args(

    # Grounder parameters
    sym_grounder_model = "ObjectCNN",
    obs_size = (64,64),
    model_name = "sym_grounder_64_fixed",

    # Environment parameters
    max_num_steps = 50,
    randomize_loc = False,
    randomize_test_loc = False,

    # Training parameters
    num_samples = 10000,
    batch_size = 32,
    seed = 1

)


train_grounder_with_agent = Args(

    # Grounder parameters
    sym_grounder_model = "ObjectCNN",
    obs_size = (64,64),
    model_name = "sym_grounder_64_fixed_w_agent",

    # Environment parameters
    max_num_steps = 50,
    randomize_loc = False,
    randomize_test_loc = False,

    # Training parameters
    num_samples = 10000,
    batch_size = 32,
    seed = 1,

    use_agent = True,
    agent_dir = "prova4"

)