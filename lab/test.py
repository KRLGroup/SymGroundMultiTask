from train_agent import Args


test = Args(
    algo="ppo",
    env="Letter-7x7-v3",
    log_interval=5,
    save_interval=20,
    frames=20000000,
    discount=0.94,
    ltl_sampler="Eventually_1_5_1_4",
    epochs=4,
    lr=0.0003
)
