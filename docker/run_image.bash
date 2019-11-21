if [ $1 = "-g" ]
then
    IMAGE=mpalm-dev-gpu
    RUN_ARGS="--net=host --runtime=nvidia bash"
else
    IMAGE=mpalm-dev-cpu2
    RUN_ARGS="--net=host bash"
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
    --volume="$HOME/mpalm_affordances/catkin_ws/src:/root/catkin_ws/src" \
    --volume="$HOME/mpalm_affordances/software:/root/software" \
    --volume="$HOME/mpalm_affordances/docker/.bashrc_new:/root/.bashrc" \
    ${IMAGE} \
    ${RUN_ARGS}


