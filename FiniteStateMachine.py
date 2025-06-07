import random
from graphviz import Source
import sys
from pythomata import SimpleDFA
from utils.deepdfa_utils import dot2pythomata, shift_back_nodes
from ltlf2dfa.parser.ltlf import LTLfParser
import numpy as np
from itertools import product


class DFA:

    # 3 types of initialization:
    # init_from_ltl: arg1 -> ltl_formula | arg2 -> num_symbols | arg3 -> formula_name
    # random_init: arg1 -> num_states | arg2 -> num_symbols
    # init_from_transacc: arg1 -> transitions | arg2 -> acceptances
    def __init__(self, arg1, arg2, arg3, dictionary_symbols = None):

        if dictionary_symbols == None:
            self.dictionary_symbols = list(range(self.num_of_symbols))
        else:
            self.dictionary_symbols = dictionary_symbols

        if isinstance(arg1, str):
            self.init_from_ltl(arg1, arg2, arg3, dictionary_symbols)
        elif isinstance(arg1, int):
            self.random_init(arg1, arg2)
        elif isinstance(arg1, dict):
            self.init_from_transacc(arg1, arg2)
        else:
            raise Exception("Uncorrect type for the argument initializing th DFA: {}".format(type(arg1)))

        self.calculate_absorbing_states()
        self.calculate_live_states()


    def calculate_absorbing_states(self):
        self.absorbing_states = []
        for q in range(self.num_of_states):
            absorbing = True
            for s in self.transitions[q].keys():
                absorbing = absorbing & (self.transitions[q][s] == q)
            if absorbing:
                self.absorbing_states.append(q)


    def calculate_live_states(self):

        self.liveliness = [self.acceptance[q] for q in range(self.num_of_states)]
        
        changed = True
        while changed:
            changed = False
            for q in range(self.num_of_states):
                if not self.liveliness[q]:
                    for s in self.transitions[q]:
                        next_q = self.transitions[q][s]
                        if self.liveliness[next_q]:
                            self.liveliness[q] = True
                            changed = True
                            break


    def init_from_ltl(self, ltl_formula, num_symbols, formula_name, dictionary_symbols, save=False):

        # convert LTL formula into DFA (dot file)
        parser = LTLfParser()
        ltl_formula_parsed = parser(ltl_formula)
        dot_dfa = ltl_formula_parsed.to_dfa()
        dot_dfa = shift_back_nodes(dot_dfa)

        # save symbolic DFA
        if save:
            with open(f"symbolicDFAs/{formula_name}_symbolic.dot", "w") as f:
                f.write(dot_dfa)
            s = Source.from_file(f"symbolicDFAs/{formula_name}_symbolic.dot")
            s.render(f"symbolicDFAs/{formula_name}_symbolic", format='pdf', cleanup=True, view=True)

        # convert dot file into SymbolicDFA
        dfa = dot2pythomata(dot_dfa, dictionary_symbols)

        # from symbolic DFA to simple DFA
        # print(dfa.__dict__)
        self.alphabet = dictionary_symbols
        self.transitions = self.reduce_dfa(dfa)
        # print(self.transitions)
        self.num_of_states = len(self.transitions)
        self.acceptance = []
        for s in range(self.num_of_states):
            if s in dfa._final_states:
                self.acceptance.append(True)
            else:
                self.acceptance.append(False)
        # print(self.acceptance)

        #Complete the transition function with the symbols of the environment that ARE NOT in the formula
        self.num_of_symbols = len(dictionary_symbols)
        self.alphabet = []
        for a in range(self.num_of_symbols):
            self.alphabet.append( a )
        if len(self.transitions[0]) < self.num_of_symbols:
            for s in range(self.num_of_states):
                for sym in self.alphabet:
                    if sym not in self.transitions[s].keys():
                        self.transitions[s][sym] = s

        #print("Complete transition function")
        #print(self.transitions)

        # save final DFA
        if save:
            self.write_dot_file(f"symbolicDFAs/{formula_name}.dot")
            s = Source.from_file(f"symbolicDFAs/{formula_name}.dot")
            s.render(f"symbolicDFAs/{formula_name}", format='pdf', cleanup=True, view=True)


    def reduce_dfa(self, pythomata_dfa):
        dfa = pythomata_dfa

        admissible_transitions = []
        for true_sym in self.alphabet:
            trans = {}
            for i, sym in enumerate(self.alphabet):
                trans[sym] = False
            trans[true_sym] = True
            admissible_transitions.append(trans)

        red_trans_funct = {}
        for s0 in dfa._states:
            red_trans_funct[s0] = {}
            transitions_from_s0 = dfa._transition_function[s0]
            for key in transitions_from_s0:
                label = transitions_from_s0[key]
                for sym, at in enumerate(admissible_transitions):
                    if label.subs(at):
                        red_trans_funct[s0][sym] = key

        return red_trans_funct


    def init_from_transacc(self, trans, acc):
        self.num_of_states = len(acc)
        self.num_of_symbols = len(trans[0])
        self.transitions = trans
        self.acceptance = acc

        self.alphabet = []
        for a in range(self.num_of_symbols):
            self.alphabet.append( a )


    def random_init(self, numb_of_states, numb_of_symbols):
        self.num_of_states = numb_of_states
        self.num_of_symbols = numb_of_symbols
        transitions= {}
        acceptance = []
        for s in range(numb_of_states):
            trans_from_s = {}
            #Each state is equiprobably set to be accepting or rejecting
            acceptance.append(bool(random.randrange(2)))
            #evenly choose another state from [i + 1; N ] and adds a random-labeled transition
            if s < numb_of_states - 1:
                s_prime = random.randrange(s + 1 , numb_of_states)
                a_start = random.randrange(numb_of_symbols)

                trans_from_s[a_start] = s_prime
            else:
                a_start = None
            for a in range(numb_of_symbols):
                #a = str(a)
                if a != a_start:
                    trans_from_s[a] = random.randrange(numb_of_states)
            transitions[s] = trans_from_s.copy()

        self.transitions = transitions
        self.acceptance = acceptance
        self.alphabet = ""
        for a in range(numb_of_symbols):
            self.alphabet += str(a)


    def accepts(self, string):
        if string == '':
            return self.acceptance[0]
        return self.accepts_from_state(0, string)


    def accepts_from_state(self, state, string):
        assert string != ''

        a = string[0]
        next_state = self.transitions[state][a]

        if len(string) == 1:
            return self.acceptance[next_state]

        return self.accepts_from_state(next_state, string[1:])


    def to_pythomata(self):
        trans = self.transitions
        acc = self.acceptance
        #print("acceptance:", acc)
        accepting_states = set()
        for i in range(len(acc)):
            if acc[i]:
                accepting_states.add(i)

        automaton = SimpleDFA.from_transitions(0, accepting_states, trans)

        return automaton


    def to_dot_str(self):
        dot_str = (
            "digraph MONA_DFA {\n"
            "rankdir = LR;\n"
            "center = true;\n"
            "size = \"7.5,10.5\";\n"
            "edge [fontname = Courier];\n"
            "node [height = .5, width = .5];\n"
            "node [shape = doublecircle];"
        )
        for i, rew in enumerate(self.acceptance):
            if rew:
                dot_str += str(i) + ";"
        dot_str += (
            "\nnode [shape = circle]; 0;\n"
            "init [shape = plaintext, label = \"\"];\n"
            "init -> 0;\n"
        )

        for s in range(self.num_of_states):
            for a in range(self.num_of_symbols):
                s_prime = self.transitions[s][a]
                dot_str += "{} -> {} [label=\"{}\"];\n".format(s, s_prime, self.dictionary_symbols[a])

        dot_str += "}\n"
        return dot_str


    def write_dot_file(self, file_name):
        with open(file_name, "w") as f:
            f.write(self.to_dot_str())


    def show(self, save_path=None):
        dot_dfa = self.to_dot_str()
        s = Source(dot_dfa)
        s.render(save_path, format='pdf', cleanup=True, view=True)


class MooreMachine(DFA):

    def __init__(self, arg1, arg2, arg3, reward = "distance", dictionary_symbols = None):
        super().__init__(arg1, arg2, arg3, dictionary_symbols)

        # (delta_o) rewards associated to each state in the MooreMachine
        self.rewards = [100 for _ in range(self.num_of_states)]

        # associate reward based on the distance from a "final state"
        if reward == "distance":

            # starts with 100 on final states, 0 otherwise
            for s in range(self.num_of_states):
                if self.acceptance[s]:
                    self.rewards[s] = 0
            #print(self.rewards)

            # propagate with fixpoint algorithm
            old_rew = self.rewards.copy()
            termination = False
            while not termination:
                termination = True
                for s in range(self.num_of_states):
                    if not self.acceptance[s]:
                        next = [ self.rewards[self.transitions[s][sym]] for sym in self.alphabet if self.transitions[s][sym] != s]
                        if len(next) > 0:
                            self.rewards[s] = 1 + min(next)

                termination = (str(self.rewards) == str(old_rew))
                old_rew = self.rewards.copy()

            for i in range(len(self.rewards)):
                self.rewards[i] *= -1
            minimum = min([r for r in self.rewards if r != -100])

            for i,r in enumerate(self.rewards):
                if r != -100:
                    self.rewards[i] = (r - minimum)

            # rescale to have maximum to 100
            maximum = max(self.rewards)
            #max : 100 = rew : x
            #x = 100 * rew / max
            for i,r in enumerate(self.rewards):
                if r != -100:
                    self.rewards[i] = 100 * r / maximum
            print("REWARDS:", self.rewards)
            #assert False

        # binary reward: 1 for "final state", 0 otherwise
        elif reward == "acceptance":
            for s in range(self.num_of_states):
                if self.acceptance[s]:
                    self.rewards[s] = 1
                else:
                    self.rewards[s] = 0

        # ternary reward: 1 for "final state", -1 for dead states, 0 otherwise
        elif reward == "ternary":
            for s in range(self.num_of_states):
                if self.acceptance[s]:
                    self.rewards[s] = 1
                elif not self.liveliness[s]:
                    self.rewards[s] = -1
                else:
                    self.rewards[s] = 0

        else:
            raise Exception("Reward based on '{}' NOT IMPLEMENTED".format(reward))


    # returns the last reward produced by the MooreMachine for the input string
    def process_trace(self, string, state=0):
        a = string[0]
        next_state = self.transitions[state][a]

        if len(string) == 1:
            return next_state, self.rewards[next_state]

        return self.process_trace(string[1:], next_state)