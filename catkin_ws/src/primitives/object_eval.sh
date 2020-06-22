export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
python evaluate_objects.py --object \
    --primitive grasp \
    --object_name basic_book \
    --experiment_name shelf_demo \
    --sim_step_repeat 20 \
    --np_seed 430 \
    --num_obj_samples 50 \
    --num_blocks 50 \
    -v \
    --goal_viz \
    --trimesh_viz
