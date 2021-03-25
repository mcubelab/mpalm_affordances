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
# IMAGE=anthonysimeonov/mpalm-dev-gpu:0.1.2
# IMAGE=anthonysimeonov/mpalm-dev-pytorch:0.1.2
IMAGE=mpalm-dev-pytorch-geom:latest
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

docker run --rm -it \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --volume="$PWD/../catkin_ws/src/primitives:/root/catkin_ws/src/primitives" \
    --volume="$PWD/../catkin_ws/src/abb_robotnode:/root/catkin_ws/src/abb_robotnode" \
    --volume="$PWD/../catkin_ws/src/config:/root/catkin_ws/src/config" \
    --volume="$PWD/../catkin_ws/src/hrl-kdl:/root/catkin_ws/src/hrl-kdl" \
    --volume="$PWD/../catkin_ws/src/ik:/root/catkin_ws/src/ik" \
    --volume="$PWD/../catkin_ws/src/realsense:/root/catkin_ws/src/realsense" \
    --volume="$PWD/../catkin_ws/src/task_planning:/root/catkin_ws/src/task_planning" \
    --volume="$PWD/workspace.sh:/workspace.sh" \
    --volume="$HOME/repos/research/airobot:/airobot" \
    --volume="$PWD/../training/:/root/training/" \
    -p 9999:9999 \
    --net=host \
    --privileged \
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
