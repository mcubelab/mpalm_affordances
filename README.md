# Docker
(on host) Go to `/path/to/mpalm_affordances/docker`

### Build
```
python docker_build.py --pytorch_geom
```

### Run
```
./run_image_user.sh
```

### Enter container
Get `$CONTAINER_NAME` from `sudo docker ps`
```
sudo docker exec -it $CONTAINER_NAME bash
```

# Setup Trained Models
(all below here, in container)

Source Python 3.6 environment, 
```
source /environments/py36/bin/activate
```

Go to `~/training/gat`, and run
```
cd ~/training/gat
./run_model_predictions.sh $SKILL
```

where `$SKILL` should be either `grasp`, `pull`, or `push`. Model paths to use for each skill are specified by hand in the `run_model_predictions.sh` file -- change them if you want to run a different model. This must be done for all the models you want to run -- `tumx` can be useful to run many of them together inside the container.

# Run simulator
Go to `~/catkin_ws/src/primitives`, and run...

### Single-step experiments
```
cd ~/catkin_ws/src/primitives
./subgoal_eval.sh
```


### Multi-step Planning Experiments
```
cd ~/catkin_ws/src/primitives
./multistep_eval.sh
```

### Explore the environment with learned models
```
cd ~/catkin_ws/src/primitives
python play_manager.py --sim
```
