#!/bin/bash
set -e

# setup ros environment
source "$HOME/catkin_ws/devel/setup.bash"
./workspace.sh

source "$HOME/catkin_ws/devel/setup.bash"
roslaunch config yumi_moveit.launch

eval "bash"

exec "$@"
