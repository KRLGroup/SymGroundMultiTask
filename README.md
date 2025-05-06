# SymGroundMultiTask

## Installing requirements (conda)

1. create a new conda env with python 3.7

    `conda create -n symgroundmultitask python=3.7`

2. install all packages from `requirements_ltl2action.txt` with

    `pip install -r requirements_ltl2action.txt`

3. install Spot 2.9 with

    `conda install -c conda-forge spot=2.9`

4. install some libraries missing from ltl2action:

    `pip install ring pygame pythomata flloat graphviz torchvision`

5. install CUDA version of dgl (GNN library, from https://anaconda.org/dglteam/dgl-cuda10.2/files?page=6) with:

    `conda install dgl-cuda10.2-0.4.3post2-py37_0.tar.bz2`

6. install CUDA 10.2 toolkit (needed by CUDA version of dgl)

    `conda install cudatoolkit=10.2 -c pytorch`

