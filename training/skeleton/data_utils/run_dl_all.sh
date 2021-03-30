source ~/environments/py36-gnn/bin/activate
### Change SAVE_DIR to be whatever local directory you wish to save the data
SAVE_DIR="${PWD}/../data"

### Don't change below, URLs link to dropbox data folder
DIR_NAMES=()
DIR_NAMES+=("skeleton_policy_dev_aug.tar.gz")
DIR_NAMES+=("skeleton_samples_only_start_goal.tar.gz")

URL_NAMES=()
URL_NAMES+=("https://www.dropbox.com/s/15c7azk6p2ebu1g/skeleton_policy_dev_augmentations_0.tar.gz?dl=0")
URL_NAMES+=("https://www.dropbox.com/s/px10wunhl79b7yb/skeleton_samples_only_start_goal.tar.xz?dl=0" )
for i in ${!DIR_NAMES[@]};
do
    echo $i
    echo ${DIR_NAMES[$i]}
    if [ -z "$1" ]
    then
        python download_data.py \
            --data_save_dir ${SAVE_DIR} \
            --data_dir_name ${DIR_NAMES[$i]}\
            --url ${URL_NAMES[$i]} 
    elif [ $1 == "d" ]  # dry run
    then
        python download_data.py \
            --data_save_dir ${SAVE_DIR} \
            --data_dir_name ${DIR_NAMES[$i]}\
            --url ${URL_NAMES[$i]} \
            -d  
    fi
done
