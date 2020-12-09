IMAGE=$USER-mpalm-dev-pytorch-geom
USERNAME=$USER
sudo docker run -it \
    --volume="$PWD/../catkin_ws/src/:/home/${USERNAME}/catkin_ws/src/" \
    --volume="$PWD/../training/:/home/${USERNAME}/training/" \
    --user="${USERNAME}" \
    --gpus all \
    ${IMAGE}
