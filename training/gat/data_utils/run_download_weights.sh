source ~/environments/py36-gnn/bin/activate
### Change SAVE_DIR to be whatever local directory you wish to save the data
SAVE_DIR="${PWD}/../vae_cachedir"

### Don't change below, URLs link to dropbox data folder
DIR_NAME="joint_gat_grasp_mask_trans_cuboid_09_09_0_dgl.tar.gz"
URL_NAME="https://www.dropbox.com/s/4o3y1pkyk3d1fo6/joint_gat_grasp_mask_trans_cuboid_09_09_0_dgl.tar.gz?dl=0"
#DIR_NAME="joint_pulling_yaw_centered_1_dgl.tar.gz"
#URL_NAME="https://www.dropbox.com/s/lseha5bkhd7uh8t/joint_pulling_yaw_centered_1_dgl.tar.gz?dl=0" 
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
