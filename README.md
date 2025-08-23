# SymGroundMultiTask

## Summary


## Installation

1. Create a new conda environment with Python 3.7.16 and the dependencies specified in environment.yml and requirements.txt:

    ```bash
    conda env create -f environment.yml
    ```

2. (optional) Install MONA if you need to create new automata:

    ```bash
    sudo apt install -y mona
    ```


## Training

1. (optional) Pretrain the GNN using the configuration in ltl_bootcamp_config.py

    ```bash
    python -m lab.run_ltl_bootcamp.py --device <device>
    ```

2. Train the Agent using the configuration in train_agent_config.py

    ```bash
    python -m lab.run_train_agent.py --device <device>
    ```


## Evaluation