### Change SAVE_DIR to be whatever local directory you wish to save the data
SAVE_DIR="${PWD}/data"

### Don't change below, URLs link to dropbox data folder
#DIR_NAME="skeleton_policy_dev_aug.tar.gz"
#URL_NAME="https://www.dropbox.com/s/15c7azk6p2ebu1g/skeleton_policy_dev_augmentations_0.tar.gz?dl=0"
DIR_NAME="skeleton_samples_only_start_goal.tar.gz"
URL_NAME="https://www.dropbox.com/s/px10wunhl79b7yb/skeleton_samples_only_start_goal.tar.xz?dl=0" 
if [ -z "$1" ]
then
    python download_data.py \
        --data_save_dir ${SAVE_DIR} \
        --data_dir_name ${DIR_NAME}\
        --url ${URL_NAME} 
elif [ $1 == "d" ]  # dry run
then
    python download_data.py \
        --data_save_dir ${SAVE_DIR} \
        --data_dir_name ${DIR_NAME}\
        --url ${URL_NAME} \
        -d  
fi
