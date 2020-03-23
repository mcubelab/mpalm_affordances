#!/bin/bash
set -e

# setup ros environment
source "$HOME/catkin_ws/devel/setup.bash"

eval "bash"

exec "$@"
