# if [ $1 = "-g" ]
# then
#     IMAGE=anthonysimeonov/mpalm-dev-gpu:0.1.2
#     RUN_ARGS="--runtime=nvidia"
# elif [ $1 = "-p" ]
# then
#     IMAGE=anthonysimeonov/mpalm-dev-pytorch:0.1.2
#     RUN_ARGS="--net=host --runtime=nvidia"    
# else
#     IMAGE=mpalm-dev-cpu
#     RUN_ARGS="--net=host"
# fi

# IMAGE=mpalm-dev-cpu
# IMAGE=mpalm-dev-gpu
IMAGE=anthonysimeonov/mpalm-dev-gpu:0.1.2
RUN_ARGS="--runtime=nvidia"

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

docker run -it \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --volume="$PWD/../catkin_ws/src/:/root/catkin_ws/src/" \
    --volume="$PWD/workspace.sh:/workspace.sh" \
    --volume="/home/anthony/repos/research/airobot:/airobot" \
    --volume="$PWD/../training/:/root/training/" \
    ${RUN_ARGS} \
    ${IMAGE}

# docker run -it \
#     --env="DISPLAY" \
#     --env="QT_X11_NO_MITSHM=1" \
#     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#     --env="XAUTHORITY=$XAUTH" \
#     --volume="$XAUTH:$XAUTH" \
#     --volume="$PWD/../catkin_ws/src/:/$HOME/catkin_ws/src/" \
#     --volume="$PWD/workspace.sh:/workspace.sh" \
#     --volume="/home/anthony/repos/research/airobot:$HOME/airobot" \
#     --volume="$PWD/../training/:/$HOME/training/" \
#     --user="anthony" \
#     ${RUN_ARGS} \
#     ${IMAGE}
