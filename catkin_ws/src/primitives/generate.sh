#!/bin/bash

source ~/catkin_ws/devel/setup.bash
export CODE_BASE=$HOME

# python generate_data.py --object --primitive grasp --num_trials 10000 \
#     --experiment_name grasping_multi_diverse_0 --sim_step_repeat 20 --np_seed 413 \
#     --num_obj_samples 500 --num_blocks 40 --multi --save_data

python generate_data.py --object \
    --primitive push \
    --experiment_name pushing_cuboid_init_1 \
    --sim_step_repeat 20 \
    --np_seed 10 \
    --num_obj_samples 5 \
    --num_blocks 40 \
    --start_trials 10 \
    --start_success 10 \
    --multi \
    --save_data \
    --save_check full \
    --shape_type cuboid