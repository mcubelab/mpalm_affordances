#IMAGE=mpalm-dev-cpu
#IMAGE=mpalm-dev-gpu
IMAGE=mpalm-dev-pytorch-geom
#IMAGE=anthonysimeonov/mpalm-dev-gpu:0.2.1
#IMAGE=anthonysimeonov/mpalm-dev-pytorch-geom:0.0.1
sudo docker run --rm -it \
    --volume="$PWD/../catkin_ws/src/:/home/asimeono/catkin_ws/src/" \
    --volume="$PWD/workspace.sh:/workspace.sh" \
    --volume="$PWD/../training/:/home/asimeono/training/" \
    --volume="$PWD/../../airobot/:/home/asimeono/airobot/" \
    --user="asimeono" \
    --gpus all \
    ${IMAGE}
