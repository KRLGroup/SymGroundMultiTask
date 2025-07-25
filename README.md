# SymGroundMultiTask

## Summary

## Creating the Environment (with conda)

1. Create a new conda environment with Python 3.7 and activate it:

    ```bash
    conda create -n symgroundmultitask python=3.7 -y
    conda activate symgroundmultitask
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt -y
    conda install -c conda-forge spot=2.9 -y
    conda install -c dglteam dgl-cuda10.2=0.4.3post2 -y
    conda install -c pytorch cudatoolkit=10.2 -y
    ```

3. (optional) Install MONA if you need to create new automata:

    ```bash
    sudo apt install -y mona
    ```