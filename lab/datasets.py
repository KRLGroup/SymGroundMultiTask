"""
Example saving formulas:
    $ python -m lab.datasets lab/datasets/e54 formulas

Example saving automata:
    $ python -m lab.datasets lab/datasets/e54 automata
"""

from pathlib import Path
from dataset import Dataset


# Available datasets

train_dataset = Dataset(
    path=Path("lab/storage/datasets/e54"),
    seed=42,
    n_formulas=10000,
    propositions=["a", "b", "c", "d", "e"],
    sampler="Eventually_1_5_1_4",
    disjoint_from=None,
)

test_dataset = Dataset(
    path=Path("lab/storage/datasets/e54test"),
    seed=42,
    n_formulas=1000,
    propositions=["a", "b", "c", "d", "e"],
    sampler="Eventually_1_5_1_4",
    disjoint_from=train_dataset,
)

hard_test_dataset = Dataset(
    path=Path("lab/storage/datasets/e65test"),
    seed=42,
    n_formulas=1000,
    propositions=["a", "b", "c", "d", "e"],
    sampler="Eventually_6_6_5_5",
    disjoint_from=None,
)

datasets = [
    train_dataset,
    test_dataset,
    hard_test_dataset
]


def get_dataset(path: Path) -> Dataset:
    for dataset in datasets:
        if dataset.path == path:
            return dataset
    raise ValueError(f"Dataset {path} not found.")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="lab/storage/datasets/e54")
    parser.add_argument("--target", type=str, default="formulas", choices=["formulas", "automata", "both"])
    args = parser.parse_args()

    dataset_path = Path(args.path)
    target = args.target
    dataset = get_dataset(dataset_path)

    if target == 'formulas' or target == 'both':
        dataset.save_formulas()
    elif target == 'automata' or target == 'both':
        dataset.save_automata()
    else:
        print(f"Unknown target {target}, expected 'formulas', 'automata' or 'both'.")