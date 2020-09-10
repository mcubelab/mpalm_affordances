export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
python evaluate_subgoal.py --object \
    --primitive grasp \
    --experiment_name grasping_uniform_project_fixed_0 \
    --sim_step_repeat 20 \
    --np_seed 435 \
    --num_obj_samples 1 \
    --num_blocks 50 \
    --object_name realsense_box