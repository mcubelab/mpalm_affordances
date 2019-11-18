#!/bin/bash
set -e

# setup ros environment
source "/home/anthony/mpalm_affordances/catkin_ws/devel/setup.bash"

eval "bash"

exec "$@"
