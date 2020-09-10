export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
python evaluate_planning.py \
    --multi \
    --experiment_name test_no_ps_pgp \
    --sim_step_repeat 20 \
    --np_seed 63 \
    --num_obj_samples 1 \
    --num_blocks 50 \
    --playback_num 1 \
    --skeleton pgp

