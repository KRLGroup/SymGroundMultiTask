# SymGroundMultiTask

## Summary

This project extends the [LTL2Action](https://github.com/LTL2Action/LTL2Action) framework to train Reinforcement Learning (RL) agents that can follow multiple temporally extended tasks expressed in Linear Temporal Logic (LTL) without requiring access to the environment's labelling function. This is done through the usage of [Neural Reward Machines](https://github.com/KRLGroup/NeuralRewardMachines), which enable to provide an indirect supervision signal to a grounder module neural network from the comparison between the ground-truth reward signals and the expected reward signals using the predicted symbols.


## Installation

1. Create a new conda environment with Python 3.7.16 and the dependencies specified in ```environment.yml``` and ```requirements.txt```:

    ```bash
    conda env create -f environment.yml
    ```

2. (optional) Install MONA if you need to create new automata:

    ```bash
    sudo apt install -y mona
    ```

3. (optional) Install Safety-Gym Environment (requires mujoco 2.1.0):

    ```bash
    pip install -e envs/safety/safety-gym/
    ```


## Dataset Creation

Create the datasets of formulas and automata needed for training the grounder:

```bash
python -m datasets.create_datasets --path <dataset>
```


## Training

1. (optional) Pretrain the GNN using the configuration in ```ltl_bootcamp_config.py```:

    ```bash
    python -m lab.run_ltl_bootcamp --device <device>
    ```

2. (optional) Pretrain the grounder using the configuration in ```train_grounder_config.py```:

    ```bash
    python -m lab.run_train_grounder --device <device>
    ```

3. Train the agent using the configuration in ```train_agent_config.py```:

    ```bash
    python -m lab.run_train_agent --device <device>
    ```


## Evaluation

1. Evaluate the grounder:

    ```bash
    python test_grounder.py --model_dir <model_name> --device <device>
    ```

2. Evaluate the agent:

    ```bash
    python test_agent.py --model_dir <model_name> --device <device>
    ```

3. Visualize the agent playing in the environment:

    ```bash
    python visualize_agent.py --model_dir <model_name> --device <device>
    ```