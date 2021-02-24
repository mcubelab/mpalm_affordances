if [ -z "$1" ] || [ $1 == 'dgl' ]
then
    GNN_LIB='dgl'
elif [ $1 == 'pyg' ]
then
    GNN_LIB='pyg'
fi
IMAGE=$USER-mpalm-dev-pytorch-$GNN_LIB
USERNAME=$USER

if [ -z "$2" ] || [ $2 == '--not_root' ]
then
    docker run --rm -it \
        --volume="$PWD/../catkin_ws/src/:/home/${USERNAME}/catkin_ws/src/" \
        --volume="$PWD/../training/:/home/${USERNAME}/training/" \
        --user="${USERNAME}" \
        --gpus all \
        -p 9999:9999 \
        -p 6006:6006 \
        --net=host \
        ${IMAGE}
elif [ $2 == '--root' ]
then
    sudo docker run --rm -it \
        --volume="$PWD/../catkin_ws/src/:/home/${USERNAME}/catkin_ws/src/" \
        --volume="$PWD/../training/:/home/${USERNAME}/training/" \
        --user="${USERNAME}" \
        --gpus all \
        -p 9999:9999 \
        -p 6006:6006 \
        --net=host \
        ${IMAGE}
fi