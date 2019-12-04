if [ $1 = "-g" ]
then
    IMAGE=anthonysimeonov/mpalm-dev-gpu:0.1.0
    RUN_ARGS="--net=host --runtime=nvidia"
else
    IMAGE=mpalm-dev-cpu
    RUN_ARGS="--net=host"
fi

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
    --volume="$HOME/repos/research/mpalms/:/mpalms" \
    --volume="$HOME/repos/research/airobot:/airobot" \
    ${RUN_ARGS} \
    ${IMAGE}
