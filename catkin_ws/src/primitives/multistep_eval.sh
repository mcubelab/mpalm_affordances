export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
python evaluate_planning.py \
    --multi \
    --experiment_name test_failure_modes_pgp_0 \
    --sim_step_repeat 20 \
    --np_seed 63 \
    --num_obj_samples 1 \
    --num_blocks 50 \
    --playback_num 1 \
    --skeleton pgp \
    --goal_viz \
    -v \
    --ignore_physics

    # --pcd_noise \
    # --pcd_noise_std 0.001 \
    # --pcd_noise_rate 0.00025 \
    # --save_data    

