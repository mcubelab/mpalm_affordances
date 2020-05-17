#!/bin/bash

source ~/catkin_ws/devel/setup.bash
export CODE_BASE=$HOME

# python generate_data.py --object --primitive grasp --num_trials 10000 \
#     --experiment_name grasping_multi_diverse_0 --sim_step_repeat 20 --np_seed 413 \
#     --num_obj_samples 500 --num_blocks 40 --multi --save_data

python generate_data.py --object --primitive pull --num_trials 10000 \
    --experiment_name pulling_multi_diverse_0 --sim_step_repeat 20 --np_seed 97 \
    --num_obj_samples 500 --num_blocks 40 --multi -v -re --save_data
