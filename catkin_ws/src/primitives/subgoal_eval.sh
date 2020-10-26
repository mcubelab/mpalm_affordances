export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
python evaluate_subgoal.py --object \
    --primitive grasp \
    --experiment_name grasping_gat_joint_depth_noise_3 \
    --sim_step_repeat 20 \
    --np_seed 135 \
    --num_obj_samples 2 \
    --num_blocks 50 \
    --object_name realsense_box \
    -v \
    --goal_viz \
    --multi \
    --save_data \
    --pcd_noise \
    --pcd_noise_std 0.0025 \
    --pcd_noise_rate 0.000025 \
    --save_data \
    --multi
