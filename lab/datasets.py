"""
Example saving formulas:
    $ python -m lab.datasets lab/datasets/e54 formulas

Example saving automata:
    $ python -m lab.datasets lab/datasets/e54 automata
"""

from pathlib import Path
from dataset import Dataset


datasets = [
    Dataset(
        path=Path("lab") / "storage" / "datasets" / "e54",
        seed=42,
        n_formulas=10000,
        propositions=["a", "b", "c", "d", "e"],
        sampler="Eventually_1_5_1_4",
        disjoint_from=None,
    ),
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
    parser.add_argument("--target", type=str, default="formulas", choices=["formulas", "automata"])
    args = parser.parse_args()


    dataset_path = Path(args.path)
    target = args.target # 'formulas' or 'automata'
    dataset = get_dataset(dataset_path)

    if target == 'formulas':
        dataset.save_formulas()
    elif target == 'automata':
        dataset.save_automata()
    else:
        print(f"Unknown target {target}, expected 'formulas' or 'automata'.")

