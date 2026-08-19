<h1 align="center">SymGroundMultiTask</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2602.09761"><img src="https://img.shields.io/badge/arXiv-2602.09761-b31b1b.svg" alt="Paper"></a>
    <a href="https://github.com/KRLGroup/SymGroundMultiTask"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href="https://rlj.cs.umass.edu/2026/papers/Paper149.html"><img src="https://img.shields.io/badge/Website-grey?logo=google-chrome&logoColor=white" alt="Website"></a>
</p>


This repository refers to the work [Grounding LTL Tasks in Sub-Symbolic RL Environments for Zero-Shot Generalization](https://arxiv.org/abs/2602.09761) presented at the main track of [Reinforcement Learning Conference (RLC) 2026](https://rl-conference.cc).


## Abstract

In this work we address the problem of training a Reinforcement Learning agent to follow multiple temporally-extended instructions expressed in Linear Temporal Logic in sub-symbolic environments. Previous multi-task work has mostly relied on knowledge of the mapping between raw observations and symbols appearing in the formulae. We drop this unrealistic assumption by jointly training a multi-task policy and a symbol grounder with the same experience. The symbol grounder is trained only from raw observations and sparse rewards via Neural Reward Machines in a semi-supervised fashion. Experiments on vision-based environments show that our method achieves performance comparable to using the true symbol grounding and significantly outperforms the only other previous method for multi-task learning that does not assume knowledge of the true symbol grounding.


## Summary

This project extends the [LTL2Action](https://github.com/LTL2Action/LTL2Action) framework to train Reinforcement Learning (RL) agents that can follow multiple temporally extended tasks expressed in Linear Temporal Logic (LTL) without requiring access to the environment's labelling function. This is done through the usage of [Neural Reward Machines](https://github.com/KRLGroup/NeuralRewardMachines), which enable to provide an indirect supervision signal to a grounder module neural network from the comparison between the ground-truth reward signals and the expected reward signals using the predicted symbols.


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/KRLGroup/SymGroundMultiTask
    ```


2. Create a new conda environment with Python 3.7.16 and the dependencies specified in ```environment.yml``` and ```requirements.txt```:

    ```bash
    cd ./SymGroundMultiTask
    conda env create -f environment.yml
    conda activate symgroundmultitask
    ```

3. (optional) Install MONA if you need to create new automata:

    ```bash
    sudo apt install -y mona
    ```

4. (optional) Replace `LTLf2DFA` with its parallelizable version to create automata more efficently:

    ```bash
    pip uninstall ltlf2dfa
    git clone https://github.com/matteopannacci/multi-LTLf2DFA.git
    pip install ./multi-LTLf2DFA
    ```

5. (optional) Install Safety-Gym Environment (requires mujoco 2.1.0):

    ```bash
    pip install -e envs/safety/safety-gym/
    ```


## Dataset Creation

Create the datasets of formulas and automata needed for training the grounder:

```bash
python -m datasets.create_datasets --name <dataset> --workers <num_workers>
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


## Citation

    @article{pannacci2026grounding,
        title={Grounding LTL Tasks in Sub-Symbolic RL Environments for Zero-Shot Generalization},
        author={Matteo Pannacci and Andrea Fanti and Elena Umili and Roberto Capobianco},
        journal={Reinforcement Learning Journal},
        volume={7},
        pages={},
        year={2026}
    }
