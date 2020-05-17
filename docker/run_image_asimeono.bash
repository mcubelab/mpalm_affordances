#IMAGE=mpalm-dev-cpu
IMAGE=mpalm-dev-gpu
#IMAGE=anthonysimeonov/mpalm-dev-gpu:0.2.1
sudo docker run -it \
    --volume="$PWD/../catkin_ws/src/:/home/asimeono/catkin_ws/src/" \
    --volume="$PWD/workspace.sh:/workspace.sh" \
    --volume="$PWD/../training/:/home/asimeono/training/" \
    --volume="$PWD/../../airobot/:/home/asimeono/airobot/" \
    --user="asimeono" \
    ${IMAGE}
