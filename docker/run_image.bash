IMAGE=anthonysimeonov/yumi-afford-dev:latest
# IMAGE=anthonysimeonov/airobot-cpu-dev:0.1.0

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
    --volume="/home/anthony/repos/research/airobot/:/home/anthony/airobot/" \
    --volume="${PWD}/../catkin_ws/src/:/root/catkin_ws/src/" \
    --runtime=nvidia \
    --net=host \
    ${IMAGE} \
    bash

# docker run -it \
#     --env="DISPLAY" \
#     --env="QT_X11_NO_MITSHM=1" \
#     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#     --env="XAUTHORITY=$XAUTH" \
#     --volume="$XAUTH:$XAUTH" \
#     --volume="${PWD}/../catkin_ws/src/:/root/catkin_ws/src/" \
#     --net=host \
#     ${IMAGE} \
#     bash