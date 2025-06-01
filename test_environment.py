import torch
import argparse
import utils
import cv2

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="GridWorld-v0")
parser.add_argument("--input_type", default="keyboard", choices=["keyboard", "terminal"])
args = parser.parse_args()


env = utils.make_env(args.env, progression_mode="full", ltl_sampler="None", seed=1, intrinsic=0, noLTL=False, device=device)
input_type = args.input_type
str_to_action = {"s":0,"d":1,"w":2,"a":3}

obs = env.reset()
done = False
step = 0

while not done:

    step += 1
    env.show()

    print(f"\n---")
    print(f"Step: {step}")

    task = obs['text']
    for key in env.symbol_to_meaning:
        task = task.replace(f"'{key}'", env.symbol_to_meaning(key))

    print(f"Task: {task}")


    print("\nAction: ", end="")

    if input_type == "terminal":
        a = input()
        while a not in str_to_action:
            print("invalid action...")
            print("Action: ", end="")
            a = input()

    elif input_type == "keyboard":
        a = None
        while a is None:
            key = cv2.waitKey(100)
            if key == 81 or key == ord('a'):
                a = "a"
            elif key == 82 or key == ord('w'):
                a = "w"
            elif key == 83 or key == ord('d'):
                a = "d"
            elif key == 84 or key == ord('s'):
                a = "s"
        print(a)

    a = str_to_action[a]
    obs, reward, done, info = env.step(a)

    if done:
        env.show()
        print(f"Reward: {reward}")
        print("Done!")
        print("Closing...")
        break

    print(f"Reward: {reward}")

if input_type == "terminal":
    input()
elif input_type == "keyboard":
    cv2.waitKey(0)

env.close()