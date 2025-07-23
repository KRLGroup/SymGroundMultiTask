from grounder_models import CNN_grounder, GridworldClassifier, ObjectCNN


grounder_models = ["ObjectCNN", "CNN_grounder", "GridworldClassifier"]


def make_grounder(model_name, n_symbols, obs_size):

    assert model_name in grounder_models

    if model_name == "ObjectCNN":
        return ObjectCNN(obs_size, n_symbols)

    elif model_name == "CNN_grounder":
        return CNN_grounder(n_symbols)

    elif model_name == "GridworldClassifier":
        return GridworldClassifier(n_symbols)

    elif model_name == None:
        return None