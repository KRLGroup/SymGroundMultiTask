import torch
import argparse
import utils

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="GridWorld-v0",
                help="name of the environment to train on (default: GridWorld-v0)")
args = parser.parse_args()


env = utils.make_env(args.env, progression_mode="full", ltl_sampler="None", seed=1, intrinsic=0, noLTL=False, device=device)
str_to_action = {"s":0,"d":1,"w":2,"a":3}

obs = env.reset()
done = False
step = 0


while not done:

    step += 1
    env.show()

    print(f"\n---")
    print(f"Step: {step}")
    print(f"Task: {obs['text']}")

    print("\nAction: ", end="")
    a = input()
    while a not in str_to_action:
        print("invalid action...")
        print("Action: ", end="")
        a = input()
    a = str_to_action[a]

    obs, reward, done, info = env.step(a%env.action_space.n)

    if done:
        env.show()
        print(f"Reward: {reward}")
        print("Done!")
        input("Closing...")
        break

    print(f"Reward: {reward}")

env.close()