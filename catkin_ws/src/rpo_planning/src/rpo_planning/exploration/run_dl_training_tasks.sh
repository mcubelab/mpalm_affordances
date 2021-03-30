### Change SAVE_DIR to be whatever local directory you wish to save the data
source ~/environments/py36-gnn/bin/activate
RPO_DIR=$(rospack find rpo_planning)
SAVE_DIR="$RPO_DIR/src/rpo_planning/data/training_tasks"
echo ${SAVE_DIR}

### Don't change below, URLs link to dropbox data folder
DIR_NAMES=()
DIR_NAMES+=("easy_rearrangement_problems.tar.gz")
DIR_NAMES+=("medium_rearrangement_problems.tar.gz")

URL_NAMES=()
URL_NAMES+=("https://www.dropbox.com/s/ough0ukm9ogi8d0/easy_rearrangement_problems.tar.gz?dl=0")
URL_NAMES+=("https://www.dropbox.com/s/q6ava9o9xfv3ibj/medium_rearrangement_problems.tar.gz?dl=0")

echo "Obtaining weight files: "
#echo ${DIR_NAMES[@]}
for i in ${!DIR_NAMES[@]};
do 
    echo $i
    echo ${DIR_NAMES[$i]}
    if [ -z "$1" ]
    then
        #rosrun rpo_planning util_download_data.py \
        python download_data.py \
            --data_save_dir ${SAVE_DIR} \
            --data_dir_name ${DIR_NAMES[$i]}\
            --url ${URL_NAMES[$i]} 
    elif [ $1 == "d" ]  # dry run
    then
        #rosrun rpo_planning util_download_data.py \
        python download_data.py \
            --data_save_dir ${SAVE_DIR} \
            --data_dir_name ${DIR_NAMES[$i]}\
            --url ${URL_NAMES[$i]} \
            -d  
    fi
done

