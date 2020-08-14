export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
python evaluate_planning.py \
    --multi \
    --experiment_name supp_video_pg \
    --sim_step_repeat 20 \
    --np_seed 62 \
    --num_obj_samples 1 \
    --num_blocks 50 \
    --playback_num 3 \
    --skeleton pg \
    -v \
    --goal_viz

#--object_name cylinder_simplify_60 --trimesh_viz #--plotly_viz
