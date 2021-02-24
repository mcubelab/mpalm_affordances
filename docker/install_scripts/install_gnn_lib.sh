#!/bin/bash

set -euxo pipefail

if [ $GNN_LIB == 'pytorch-geometric' ]
then
    # source $HOME/environments/py36-gnn/bin/activate && \
    #     pip install --upgrade pip && \
    #     pip install torch===1.4.0 torchvision==0.5.0 && \
    #     pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-torch-1.4.0+${CUDA}.html --no-cache-dir --force-reinstall && \
    #     pip install torch-sparse==0.6.3 -f https://pytorch-geometric.com/whl/torch-torch-1.4.0+${CUDA}.html --no-cache-dir --force-reinstall && \
    #     pip install torch-cluster==1.5.5 -f https://pytorch-geometric.com/whl/torch-torch-1.4.0+${CUDA}.html --no-cache-dir --force-reinstall && \
    #     pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-torch-1.4.0+${CUDA}.html --no-cache-dir --force-reinstall && \
    #     pip install torch-geometric==1.4.3 -f https://pytorch-geometric.com/whl/torch-torch-1.4.0+${CUDA}.html --no-cache-dir && \
    #     pip install ipython
    source $HOME/environments/py36-gnn/bin/activate
    pip install --upgrade pip
    pip install torch===1.4.0 torchvision==0.5.0 ipython
    pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-torch-1.4.0+${CUDA}.html
    pip install torch-sparse==0.6.3 -f https://pytorch-geometric.com/whl/torch-torch-1.4.0+${CUDA}.html
    pip install torch-cluster==1.5.5 -f https://pytorch-geometric.com/whl/torch-torch-1.4.0+${CUDA}.html
    pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-torch-1.4.0+${CUDA}.html
    pip install torch-geometric==1.4.3
elif [ $GNN_LIB == 'deep-graph-library' ]
then
    # make virtualenv for dgl using pytorch 1.5, to be able to run either one
    source ${HOME}/environments/py36-gnn/bin/activate && \
        pip install torch==1.5.0 torchvision dgl-$CUDA
fi

source ${HOME}/environments/py36-gnn/bin/activate && \
    pip install ipython trimesh tensorboard rospkg pybullet yacs