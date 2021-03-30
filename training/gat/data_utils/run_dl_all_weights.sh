source ~/environments/py36-gnn/bin/activate
### Change SAVE_DIR to be whatever local directory you wish to save the data
SAVE_DIR="${PWD}/../vae_cachedir"

### Don't change below, URLs link to dropbox data folder
DIR_NAMES=()
DIR_NAMES+=("joint_pulling_yaw_centered_1_dgl.tar.gz")
DIR_NAMES+=("joint_gat_grasp_mask_trans_cuboid_09_09_0_dgl.tar.gz")
DIR_NAMES+=("joint_pushing_init_centered_2_dgl.tar.gz")

URL_NAMES=()
URL_NAMES+=("https://www.dropbox.com/s/lseha5bkhd7uh8t/joint_pulling_yaw_centered_1_dgl.tar.gz?dl=0") 
URL_NAMES+=("https://www.dropbox.com/s/4o3y1pkyk3d1fo6/joint_gat_grasp_mask_trans_cuboid_09_09_0_dgl.tar.gz?dl=0")
URL_NAMES+=("https://www.dropbox.com/s/pyqsd9hwoxw88qn/joint_pushing_init_centered_2_dgl.tar.gz?dl=0")

echo "Obtaining weight files: "
#echo ${DIR_NAMES[@]}
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
