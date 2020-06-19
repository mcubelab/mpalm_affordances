export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
python evaluate_subgoal.py --object \
    --primitive grasp \
    --experiment_name mustard_demo_1 \
    --sim_step_repeat 20 \
    --np_seed 430 \
    --num_obj_samples 50 \
    --num_blocks 50 \
    --object_name mustard_coarse \
    -v \
    --goal_viz \
    --final_subgoal

