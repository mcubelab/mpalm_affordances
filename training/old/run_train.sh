# python train_vae.py --model_name pose_init_small_batch_1 \
# --total_data_size 1000 --batch_size 100 --latent_dimension 2 --start_epoch 50 --num_epochs 100

# python train_vae.py --model_name grasp_pose_init_small_batch_0 \
# --total_data_size 1000 --batch_size 100 --latent_dimension 3 --num_epochs 200 \
# --data_dir /root/catkin_ws/src/primitives/data/grasp/face_ind_test_0_fixed \
# --output_dimension 14 --start_epoch 99

# python train_vae.py --model_name pull_keypoints_init_small_batch_0 \
# --total_data_size 1000 --batch_size 100 --latent_dimension 3 --num_epochs 100 \
# --input_dimension 31 --start_rep keypoints

# python train_vae.py --model_name grasp_keypoints_two_heads_full_data_grasp_input_1 \
# --total_data_size 3700 --batch_size 512 --latent_dimension 3 --num_epochs 200 --start_epoch 2 \
# --data_dir /root/catkin_ws/src/primitives/data/grasp/face_ind_test_0_fixed \
# --output_dimension 7 --input_dimension 45 --start_rep keypoints --learning_rate 1e-4

# python train_vae.py --model_name pull_keypoints_two_heads_diverse_goals_full_data_palm_input_0 \
# --total_data_size 4901 --batch_size 512 --latent_dimension 3 --num_epochs 200 \
# --data_dir /root/catkin_ws/src/primitives/data/pull/pull_face_all_0/train \
# --output_dimension 7 --input_dimension 38 --start_rep keypoints --learning_rate 1e-4 --skill_type pull

# python train_vae.py --model_name pull_keypoints_two_heads_diverse_goals_full_data_palm_input_0 \
# --total_data_size 4901 --batch_size 1028 --latent_dimension 3 --num_epochs 200 \
# --data_dir /root/catkin_ws/src/primitives/data/pull/pull_face_all_0/train \
# --output_dimension 7 --input_dimension 38 --start_rep keypoints --learning_rate 1e-4 --skill_type pull

# python train_vae.py --model_name pull_keypoints_goal_small_batch_0 \
# --total_data_size 100 --batch_size 10 --latent_dimension 2 --num_epochs 200 \
# --data_dir /root/catkin_ws/src/primitives/data/pull/pull_face_all_0/train \
# --output_dimension 7 --input_dimension 31 --start_rep keypoints --learning_rate 1e-4 --skill_type pull --task goal

# python train_vae.py --model_name pull_keypoints_start_and_goal_small_batch_0 \
# --total_data_size 100 --batch_size 10 --latent_dimension 2 --num_epochs 200 \
# --data_dir /root/catkin_ws/src/primitives/data/pull/pull_face_all_0/train \
# --output_dimension 24 --input_dimension 48 --start_rep keypoints --goal_rep keypoints \
# --learning_rate 1e-4 --skill_type pull --task goal

# python train_vae.py --model_name pull_keypoints_goal_0 \
# --total_data_size 4901 --batch_size 1028 --latent_dimension 3 --num_epochs 200 \
# --data_dir /root/catkin_ws/src/primitives/data/pull/pull_face_all_0/train \
# --output_dimension 7 --input_dimension 38 --start_rep keypoints --learning_rate 1e-4 --skill_type pull --task goal

#python train_vae.py --model_name grasp_transformation_full_data_0 \
#--total_data_size 3339 --batch_size 512 --latent_dimension 3 --num_epochs 200 \
#--data_dir /root/catkin_ws/src/primitives/data/grasp/face_ind_test_0_fixed/train \
#--output_dimension 7 --input_dimension 7 --start_rep keypoints --goal_rep keypoints \
#--learning_rate 1e-4 --skill_type pull --task transformation

# python train_vae.py --model_name pull_transformation_full_data_start_keypoints_cond_batch_512_fixed_1 \
# --total_data_size 4901 --batch_size 512 --latent_dimension 3 --num_epochs 1000 \
# --data_dir /root/catkin_ws/src/primitives/data/pull/pull_face_all_0/train \
# --output_dimension 7 --input_dimension 31 --start_rep keypoints --goal_rep keypoints \
# --learning_rate 1e-4 --skill_type pull --task transformation --pos_beta 1.0 --kl_anneal_rate 0.99999999 \
# --log_dir /root/training/runs/pull_trans_cond_start --save_freq 50

# python train_vae.py --model_name pull_transformation_full_data_start_keypoints_cond_batch_512_fixed_no_xy_0 \
# --total_data_size 4901 --batch_size 512 --latent_dimension 3 --num_epochs 1000 \
# --data_dir /root/catkin_ws/src/primitives/data/pull/pull_face_all_0/train \
# --output_dimension 7 --input_dimension 31 --start_rep keypoints --goal_rep keypoints \
# --learning_rate 1e-4 --skill_type pull --task transformation --pos_beta 1.0 --kl_anneal_rate 0.99999999 \
# --log_dir /root/training/runs/pull_trans_cond_start --save_freq 50

# python train_vae.py --model_name pull_keypoints_goal_full_data_start_keypoints_cond_batch_128_anneal_3 \
# --total_data_size 4901 --batch_size 128 --latent_dimension 3 --num_epochs 500 \
# --data_dir /root/catkin_ws/src/primitives/data/pull/pull_face_all_0/train \
# --output_dimension 7 --input_dimension 31 --start_rep keypoints --goal_rep keypoints \
# --learning_rate 1e-4 --skill_type pull --task goal --pos_beta 1.0 --kl_anneal_rate 0.9999999 \
# --log_dir /root/training/runs/pull_goal_cond_start

# python train_vae.py --model_name grasp_transformation_cond_start_full_data_fixed_anneal_0 \
# --total_data_size 3339 --batch_size 128 --latent_dimension 3 --num_epochs 1000 --start_epoch 0 \
# --data_dir /root/catkin_ws/src/primitives/data/grasp/face_ind_test_0_fixed/train \
# --output_dimension 7 --input_dimension 31 --start_rep keypoints --goal_rep keypoints \
# --learning_rate 1e-4 --skill_type grasp --task transformation --kl_anneal_rate 0.9999999 \
# --log_dir /root/training/runs/grasp_trans_cond_start --save_freq 20

#python train_vae.py --model_name grasp_keypoints_goal_full_data_anneal_init_128_batch_2 \
#--total_data_size 3339 --batch_size 128 --latent_dimension 3 --num_epochs 1000 --start_epoch 0 \
#--data_dir /root/catkin_ws/src/primitives/data/grasp/face_ind_test_0_fixed/train \
#--output_dimension 7 --input_dimension 31 --start_rep keypoints --goal_rep keypoints \
#--learning_rate 1e-4 --skill_type grasp --task goal --kl_scalar 1 \
#--log_dir /root/training/runs/grasp_goal_keypoints \
#--save_freq 20 --kl_anneal_rate 0.999999


# python train_vae.py --model_name grasp_contact_keypoints_start_goal_fixed_full_data_anneal_batch128_fixed \
# --total_data_size 3339 --batch_size 128 --latent_dimension 3 --num_epochs 1000 --start_epoch 0 \
# --data_dir /root/catkin_ws/src/primitives/data/grasp/face_ind_test_0_fixed/train \
# --output_dimension 7 --input_dimension 55 --start_rep keypoints --goal_rep keypoints \
# --learning_rate 1e-4 --skill_type grasp --task contact --kl_scalar 1 \
# --log_dir /root/training/runs/grasp_contact_fixed \
# --save_freq 50 --kl_anneal_rate 0.999999


python train_vae.py --model_name pull_contact_keypoints_start_goal_fixed_full_data_anneal_batch128_fixed_1 \
--total_data_size 4901 --batch_size 128 --latent_dimension 3 --num_epochs 1000 --start_epoch 0 \
--data_dir /root/catkin_ws/src/primitives/data/pull/pull_face_all_0/train \
--output_dimension 7 --input_dimension 55 --start_rep keypoints --goal_rep keypoints \
--learning_rate 1e-4 --skill_type pull --task contact --kl_scalar 1 \
--log_dir /root/training/runs/pull_contact_fixed \
--save_freq 20 --kl_anneal_rate 0.999