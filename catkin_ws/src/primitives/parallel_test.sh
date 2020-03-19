# python primitive_data_gen.py --object --num_trials 20000 \
#     --experiment_name viz --step_repeat 20 -re --np_seed 4 

# python primitive_data_gen.py --object --primitive grasp --num_trials 10000 \
#     --experiment_name viz --step_repeat 20 --np_seed 11 -v --debug --multi  

python parallel_data_gen_test.py --object --primitive grasp --num_trials 10000 \
    --experiment_name test_parallel --step_repeat 20 --np_seed 11 -v --debug --save_data
