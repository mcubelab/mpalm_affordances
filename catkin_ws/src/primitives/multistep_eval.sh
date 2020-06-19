export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
python evaluate_planning.py \
    --multi \
    --experiment_name multistep_uniform_0_pgp \
    --sim_step_repeat 20 \
    --np_seed 5 \
    --num_obj_samples 1 \
    --num_blocks 50 \
    --save_data \
    --playback_num 1 \
    --skeleton pgp \
    -v

    # --trimesh_viz
#--object_name cylinder_simplify_60 --trimesh_viz #--plotly_viz
