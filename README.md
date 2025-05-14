# SymGroundMultiTask

## Installing requirements (conda)

1. Create a new conda environment with Python 3.7:

    ```bash
    conda create -n symgroundmultitask python=3.7
    ```

2. Install all packages from `requirements_ltl2action.txt`:

    ```bash
    pip install -r requirements_ltl2action.txt
    ```

3. Install Spot 2.9:

    ```bash
    conda install -c conda-forge spot=2.9
    ```

4. Install some libraries missing from ltl2action:

    ```bash
    pip install ring pygame pythomata flloat graphviz torchvision
    ```

5. Install the CUDA version of DGL (GNN library):

    ```bash
    conda install -c dglteam dgl-cuda10.2=0.4.3post2
    ```

6. Install the CUDA 10.2 toolkit (needed by the CUDA version of DGL):

    ```bash
    conda install cudatoolkit=10.2 -c pytorch
    ```