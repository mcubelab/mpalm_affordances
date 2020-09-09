IMAGE=$USER-mpalm-dev-pytorch-geom
USERNAME=$USER
sudo docker run -it \
    --volume="$PWD/../catkin_ws/src/:/home/${USERNAME}/catkin_ws/src/" \
    --volume="$PWD/workspace.sh:/workspace.sh" \
    --volume="$PWD/../training/:/home/${USERNAME}/training/" \
    --volume="$PWD/../../airobot/:/home/${USERNAME}/airobot/" \
    --user="${USERNAME}" \
    --gpus all \
    ${IMAGE}
