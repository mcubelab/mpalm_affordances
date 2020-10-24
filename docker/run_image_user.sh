IMAGE=$USER-mpalm-dev-pytorch-geom
USERNAME=$USER
# sudo docker run -it \
#     --volume="$PWD/../catkin_ws/src/:/home/${USERNAME}/catkin_ws/src/" \
#     --volume="$PWD/../training/:/home/${USERNAME}/training/" \
#     --user="${USERNAME}" \
#     --gpus all \
#     ${IMAGE}

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
    --volume="$PWD/../catkin_ws/src/primitives:/home/${USERNAME}/catkin_ws/src/primitives" \
    --volume="$PWD/../catkin_ws/src/abb_robotnode:/home/${USERNAME}/catkin_ws/src/abb_robotnode" \
    --volume="$PWD/../catkin_ws/src/config:/home/${USERNAME}/catkin_ws/src/config" \
    --volume="$PWD/../catkin_ws/src/hrl-kdl:/home/${USERNAME}/catkin_ws/src/hrl-kdl" \
    --volume="$PWD/../catkin_ws/src/ik:/home/${USERNAME}/catkin_ws/src/ik" \
    --volume="$PWD/../catkin_ws/src/realsense:/home/${USERNAME}/catkin_ws/src/realsense" \
    --volume="$PWD/../catkin_ws/src/realsense-ros:/home/${USERNAME}/catkin_ws/src/realsense-ros" \
    --volume="$PWD/../catkin_ws/src/task_planning:/home/${USERNAME}/catkin_ws/src/task_planning" \
    --volume="$PWD/../catkin_ws/src/hand_eye_calibration:/home/${USERNAME}/catkin_ws/src/hand_eye_calibration" \
    --volume="$PWD/../catkin_ws/src/aruco_ros:/home/${USERNAME}/catkin_ws/src/aruco_ros" \
    --volume="$PWD/workspace.sh:/workspace.sh" \
    --volume="$HOME/anthony/repos/research/airobot:/airobot" \
    --volume="$PWD/../training/:/home/${USERNAME}/training/" \
    --user="${USERNAME}" \
    --net=host \
    --privileged \
    --gpus all \
    ${IMAGE}
