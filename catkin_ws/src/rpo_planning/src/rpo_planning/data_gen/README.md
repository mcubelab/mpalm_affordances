# Environment setup
## (Local)
### Build docker image
```
cd /path/to/mpalm_affordances/docker
# for dry run, add -d flag
python docker_build.py
```

### Launch docker container
```
cd /path/to/mpalm_affordances/docker
./run_image_user.sh
```
---

## (Inside Container)
Note that if you ever need to enter the container that is already running, use the command `docker exec -it $CONTAINER_NAME bash` where `$CONTAINER_NAME` can be viewed by `docker ps`

### Setup environment
```
cd ~/setup
./setup_full.sh
```

### Setup robot services (motion planning, etc.)
```
roslaunch config yumi_moveit.launch
```

# Run data generation
### Run script
```
roscd rpo_planning/src/rpo_planning/data_gen
python generate_data.py
```

### Saved data and environment visualization
By default, data will be saved in `mpalm_affordances/catkin_ws/src/rpo_planning/src/rpo_planning/data/$PRIMITIVE_NAME/$EXPERIMENT_NAME` where `$PRIMITIVE_NAME` and `$EXPERIMENT_NAME` are specified as arguments to the `generate_data.py` script. 

By default, videos will be saved in `mpalm_affordances/catkin_ws/src/rpo_planning/src/rpo_planning/data/$PRIMITIVE_NAME/$EXPERIMENT_NAME/vid`. 

These videos can be examined to tell what the robot is doing while executing its trajectories. This can help with debugging if you cannot locally visualize the PyBullet environment.

If you are able to run PyBullet in GUI mode and you want to visualize the data generation procedure, pass `-v` as a flag to `generate_data.py`.
