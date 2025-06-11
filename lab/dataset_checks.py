import argparse
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--dataset1_path", type=str, default="/home/matt/SymGroundMultiTask/lab/storage/datasets/e54/formulas.pkl")
parser.add_argument("--dataset2_path", type=str, default="/home/matt/SymGroundMultiTask/lab/storage/datasets/e54test/formulas.pkl")
args = parser.parse_args()


print("dataset1 duplicates:")
duplicates = utils.get_dataset_duplicates(dataset1_path)
print(duplicates)

print("dataset2 duplicates:")
duplicates = utils.get_dataset_duplicates(dataset2_path)
print(duplicates)

print("indexes of common formulas:")
indexes = utils.get_dataset_common_index_pairs(
    dataset1_path,
    dataset2_path
)
print(indexes)