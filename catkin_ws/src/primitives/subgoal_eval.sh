export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
python evaluate_subgoal.py --object \
    --primitive grasp \
    --experiment_name grasping_uniform_project_fixed_0 \
    --sim_step_repeat 20 \
    --np_seed 435 \
    --num_obj_samples 1 \
    --num_blocks 50 \
    --object_name realsense_box \
    --goal_viz \
    --multi \
    -v \
    --save_data

    # --resume \
    # --resume_data_dir '/root/catkin_ws/src/primitives/data/push/pushing_joint_pointnet_filtering_0_07_17_20_05-24-42/'
# '/root/catkin_ws/src/primitives/data/push/pushing_joint_local_filtering_0_07_17_20_05-24-13/'
# '/root/catkin_ws/src/primitives/data/push/pushing_joint_pointnet_filtering_0_07_17_20_05-24-42/'
# '/root/catkin_ws/src/primitives/data/push/pushing_h_local_filtering_0_07_17_20_05-23-53/'