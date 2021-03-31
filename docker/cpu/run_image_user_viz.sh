if [ -z "$1" ] || [ $1 == 'dgl' ]
then
    GNN_LIB='dgl'
elif [ $1 == 'pyg' ]
then
    GNN_LIB='pyg'
fi
### PUT WHATEVER IMAGE YOU WANT TO RUN HERE
IMAGE=$USER-mpalm-dev-pytorch-$GNN_LIB
#IMAGE=$USER-mpalm-dev-cpu

USERNAME=$USER
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

docker run --rm -it \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --volume="$PWD/../catkin_ws/src/:/home/${USERNAME}/catkin_ws/src/" \
    --volume="$PWD/../training/:/home/${USERNAME}/training/" \
    --volume="$PWD/../setup/:/home/${USERNAME}/setup/" \
    --user="${USERNAME}" \
    --gpus all \
    -p 6006:6006 \
    --net=host \
    ${IMAGE}

