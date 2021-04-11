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

cd ~/training/gat
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
python model_predictions_lcm.py --cuda \
    --primitive_name pull \
    --model_path 'joint_pulling_yaw_centered_1_dgl' \
    --model_number 20000 \
    --num_workers ${NUM_WORKERS} \
    --gnn_library ${GNN_LIB} &
pull_nn_server_proc_id=$!
pids+=($!)

# below for push GAT
python model_predictions_lcm.py --cuda \
    --primitive_name push \
    --model_path 'joint_pushing_init_centered_2_dgl' \
    --model_number 65000 \
    --num_workers ${NUM_WORKERS} \
    --gnn_library ${GNN_LIB} &
push_nn_server_proc_id=$!
pids+=($!)

# launch tensorboard
cd ~/training/skeleton; tensorboard --logdir runs --port 6007 &
tb_proc_id=$!
pids+=($!)

# launch ros
deactivate; cd ~; source ~/catkin_ws/devel/setup.bash; roslaunch config yumi_moveit.launch &
moveit_proc_id=$!
pids+=($!)

while read SIGNAL; do
    case "$SIGNAL" in
        *EXIT*)break;;
        *)echo "signal  $SIGNAL  is unsupported" >/dev/stderr;;
    esac
done < /tmp/mypipe
