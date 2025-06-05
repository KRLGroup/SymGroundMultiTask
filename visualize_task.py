import pickle
import os
import argparse
import ast
from utils import ltl2dfa

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="file", choices=["file", "manual", "sampler"])
parser.add_argument("--folder", type=str, default="envs/gridworld_multitask/tasks")
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--sampler", type=str, default="keyboard", choices=["keyboard", "terminal"])
parser.add_argument("--formula", type=ast.literal_eval, default=('eventually', 'b'))
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

    formula = tuple(args.formula)
    automaton = ltl2dfa(formula, ["a", "b", "c", "d", "e"])


elif args.mode == "sampler":
    raise NotImplementedError


print(f"Formula: {formula}")
automaton.write_dot_file(os.path.join(OUTPUT_DIR, "automaton"), show=True)