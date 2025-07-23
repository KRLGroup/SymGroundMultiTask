import pickle
import os
import argparse, ast

from utils import ltl_ast2dfa, pprint_ltl_formula
from ltl_samplers import getLTLSampler


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="file", choices=["file", "manual", "sampler"])
parser.add_argument("--folder", type=str, default="datasets/e54")
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--symbols", nargs="+", type=str, default=["a", "b", "c", "d", "e"])
parser.add_argument("--formula", type=ast.literal_eval, default=('eventually', 'b'))
parser.add_argument("--sampler", type=str, default="Eventually_1_5_1_4")
parser.add_argument('--automaton', dest='automaton', default=True, action='store_true')
parser.add_argument('--no-automaton', dest='automaton', action='store_false')
args = parser.parse_args()

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(REPO_DIR, "saves")


if args.mode == "file":
    TASKS_DIR = os.path.join(REPO_DIR, args.folder)
    with open(os.path.join(TASKS_DIR, "formulas.pkl"), "rb") as f:
        formula = pickle.load(f)[args.id]

elif args.mode == "manual":
    formula = tuple(args.formula)

elif args.mode == "sampler":
    sampler = getLTLSampler(args.sampler, args.symbols)
    formula = sampler.sample()

print("Formula:")
pprint_ltl_formula(formula)


if args.automaton:

    if args.mode == "file":
        with open(os.path.join(TASKS_DIR, "automata.pkl"), "rb") as f:
            automaton = pickle.load(f)[args.id]

    if args.mode == "manual":
        automaton = ltl_ast2dfa(formula, args.symbols)

    if args.mode == "sampler":
        automaton = ltl_ast2dfa(formula, args.symbols)

    automaton.write_dot_file(os.path.join(OUTPUT_DIR, "automaton.dot"))
    automaton.show(os.path.join(OUTPUT_DIR, "automaton"))