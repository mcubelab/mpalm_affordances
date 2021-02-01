if [ -z "$1" ]
then
    python download_data.py \
        --data_save_dir catkin_ws/src/primitives/data \
        --data_dir_name skeleton_policy_aug_test_dl.tar.gz \
        --url https://www.dropbox.com/s/15c7azk6p2ebu1g/skeleton_policy_dev_augmentations_0.tar.gz?dl=0
elif [ $1 == "d" ]
then
    python download_data.py \
        --data_save_dir catkin_ws/src/primitives/data \
        --data_dir_name skeleton_policy_aug_test_dl.tar.gz \
        --url https://www.dropbox.com/s/15c7azk6p2ebu1g/skeleton_policy_dev_augmentations_0.tar.gz?dl=0 \
        -d
fi