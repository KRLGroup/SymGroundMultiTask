import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="file", choices=["file", "manual", "sampler"])
parser.add_argument("--folder", default="envs/gridworld_multitask/tasks")
parser.add_argument("--id", default=0)
parser.add_argument("--sampler", default="keyboard", choices=["keyboard", "terminal"])
args = parser.parse_args()

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(MAIN_DIR, "saves")


if args.mode == "file":

    TASKS_DIR = os.path.join(MAIN_DIR, args.folder)

    with open(os.path.join(TASKS_DIR, "formulas.pkl"), "rb") as f:
        formula = pickle.load(f)[args.id]
    with open(os.path.join(TASKS_DIR, "automata.pkl"), "rb") as f:
        automaton = pickle.load(f)[args.id]


elif args.mode == "manual":
    raise NotImplementedError


elif args.mode == "sampler":
    raise NotImplementedError



print(formula)
automaton.write_dot_file(os.path.join(OUTPUT_DIR, "automaton"))