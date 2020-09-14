export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
python evaluate_planning.py \
    --multi \
    --experiment_name baseline_better_heuristics_pgp_goal_feas_track_samples_3 \
    --sim_step_repeat 20 \
    --np_seed 200 \
    --num_obj_samples 1 \
    --num_blocks 50 \
    --playback_num 1 \
    --skeleton pgp \
    --goal_viz \
    -v \
    --ignore_physics \
    --save_data \
    --baseline

    # --pcd_noise \
    # --pcd_noise_std 0.0025 \
    # --pcd_noise_rate 0.00025 

