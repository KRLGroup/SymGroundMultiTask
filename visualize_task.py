import os
import argparse, ast

from utils import ltl_ast2dfa, pprint_ltl_formula
from ltl_samplers import getLTLSampler


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="sampler", choices=["sampler", "manual"])
parser.add_argument("--sampler", type=str, default="Dataset_e54")
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--formula", type=ast.literal_eval, default=('eventually', 'b'))
parser.add_argument("--symbols", nargs="+", type=str, default=["a", "b", "c", "d", "e"])
parser.add_argument('--automaton', dest='automaton', default=True, action='store_true')
parser.add_argument('--no-automaton', dest='automaton', action='store_false')
args = parser.parse_args()

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(REPO_DIR, "saves")


if args.mode == "sampler":
    sampler = getLTLSampler(args.sampler, args.symbols)
    if "Dataset" in args.sampler:
        formula = sampler.get_formula(args.id)
    else:
        sampler.sample()

elif args.mode == "manual":
    formula = tuple(args.formula)


print("Formula:")
pprint_ltl_formula(formula)


if args.automaton:

    if args.mode == "sampler":
        if "Dataset" in args.sampler:
            automaton = sampler.get_automaton(args.id)
        else:
            automaton = ltl_ast2dfa(formula, args.symbols)

    if args.mode == "manual":
        automaton = ltl_ast2dfa(formula, args.symbols)

    automaton.write_dot_file(os.path.join(OUTPUT_DIR, "automaton.dot"))
    automaton.show(os.path.join(OUTPUT_DIR, "automaton"))