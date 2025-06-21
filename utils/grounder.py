from grounder_models import CNN_grounder, GridworldClassifier, ObjectCNN
from envs.gridworld_multitask.Environment import OBS_SIZE


grounder_models = ["ObjectCNN", "CNN_grounder", "GridworldClassifier"]


def make_grounder(model_name, n_symbols):

    assert model_name in grounder_models

    if model_name == "ObjectCNN":
        return ObjectCNN((OBS_SIZE,OBS_SIZE), n_symbols).double()

    elif model_name == "CNN_grounder":
        return CNN_grounder(n_symbols).double()

    elif model_name == "GridworldClassifier":
        return GridworldClassifier(n_symbols).double()

    elif model_name == None:
        return None