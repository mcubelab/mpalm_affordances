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

python train_vae.py --model_name pull_transformation_full_data_0 \
--total_data_size 4901 --batch_size 512 --latent_dimension 3 --num_epochs 200 \
--data_dir /root/catkin_ws/src/primitives/data/pull/pull_face_all_0/train \
--output_dimension 7 --input_dimension 7 --start_rep keypoints --goal_rep keypoints \
--learning_rate 1e-4 --skill_type pull --task transformation