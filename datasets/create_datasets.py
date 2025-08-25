from pathlib import Path
import os
from .dataset import Dataset


DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))


# Available datasets

train_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "e54")),
    seed = 42,
    n_formulas = 10000,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "Eventually_1_5_1_4",
    disjoint_from = None,
)

test_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "e54test")),
    seed = 42,
    n_formulas = 1000,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "Eventually_1_5_1_4",
    disjoint_from = train_dataset,
)

hard_test_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "e65test")),
    seed = 42,
    n_formulas = 50,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "Eventually_6_6_5_5",
    disjoint_from = None,
)

ga_train_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "ga432")),
    seed = 42,
    n_formulas = 10000,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "GlobalAvoidance_1_4_1_3_1_2",
    disjoint_from = None,
)

ga_test_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "ga432test")),
    seed = 42,
    n_formulas = 1000,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "GlobalAvoidance_1_4_1_3_1_2",
    disjoint_from = ga_train_dataset,
)

ga_hard_test_dataset = Dataset(
    path = Path(os.path.join(DATASETS_DIR, "ga542test")),
    seed = 42,
    n_formulas = 50,
    propositions = ["a", "b", "c", "d", "e"],
    sampler = "GlobalAvoidance_5_5_4_4_2_2",
    disjoint_from = None,
)


datasets = [

    train_dataset,
    test_dataset,
    hard_test_dataset,

    ga_train_dataset,
    ga_test_dataset,
    ga_hard_test_dataset

]


def get_dataset(path: Path) -> Dataset:
    resolved_path = path.resolve()
    for dataset in datasets:
        if dataset.path.resolve() == resolved_path:
            return dataset
    raise ValueError(f"Dataset {path} not found.")



# Warning: because of how the LTLf2DFA works, having more than one process creating automata at 
# the same time may cause inconsistencies and errors

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./datasets/e54")
    parser.add_argument("--target", type=str, default="both", choices=["formulas", "automata", "both"])
    args = parser.parse_args()

    dataset_path = Path(args.path)
    target = args.target
    dataset = get_dataset(dataset_path)

    if target == 'formulas':
        dataset.save_formulas()
    elif target == 'automata':
        dataset.save_automata()
    elif target == 'both':
        dataset.save_formulas()
        dataset.save_automata()
    else:
        print(f"Unknown target {target}, expected 'formulas', 'automata' or 'both'.")