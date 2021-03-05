if [ -z "$1" ]
then
    NUM_WORKERS=1
else
    NUM_WORKERS=${1}
fi

mkfifo /tmp/mypipe
killbg() {
    for p in "${pids[@]}" ; do
        echo 'Killing'
        kill "$p";
    done
    rm /tmp/mypipe
}
trap killbg EXIT
pids=()

export GNN_LIB=dgl
source ~/environments/py36-gnn/bin/activate

# below for grasp GAT
python model_predictions_lcm.py --cuda \
    --primitive_name grasp \
    --model_path 'joint_gat_grasp_mask_trans_cuboid_09_09_0_dgl' \
    --model_number 20000 \
    --num_workers ${NUM_WORKERS} \
    --gnn_library ${GNN_LIB} &
grasp_nn_server_proc_id=$!
pids+=($!)

# below for pull GAT
python model_predictions_lcm.py \
    --primitive_name pull \
    --model_path 'joint_pulling_yaw_centered_1_dgl' \
    --model_number 20000 \
    --num_workers ${NUM_WORKERS} \
    --gnn_library ${GNN_LIB} &
pull_nn_server_proc_id=$!
pids+=($!)

# # below for push GAT
# python model_predictions_lcm.py --cuda \
#     --primitive_name push \
#     --model_path 'joint_pushing_init_centered_2' \
#     --model_number 65000 \
#     --gnn_library $GNN_LIB


while read SIGNAL; do
    case "$SIGNAL" in
        *EXIT*)break;;
        *)echo "signal  $SIGNAL  is unsupported" >/dev/stderr;;
    esac
done < /tmp/mypipe