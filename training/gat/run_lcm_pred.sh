export GNN_LIB=dgl

# below for grasp GAT
if [ $1 == 'grasp' ]
then
    # GAT
    python model_predictions_lcm.py --cuda \
        --primitive_name grasp \
        --model_path 'joint_gat_grasp_mask_trans_cuboid_09_09_0_dgl' \
        --model_number 20000 \
        --gnn_library $GNN_LIB
    # # PointNet
    # python model_predictions.py --cuda \
    #     --primitive_name grasp \
    #     --model_path 'pointnet_joint_2' \
    #     --model_number 30000 \
    #     --pointnet
elif [ $1 == 'pull' ]
then
# # below for pull GAT
    python model_predictions_lcm.py \
        --primitive_name pull \
        --model_path 'joint_pulling_yaw_centered_1_dgl' \
        --model_number 20000 \
        --gnn_library $GNN_LIB
elif [ $1 == 'push' ]
then
# # below for push GAT
    python model_predictions_lcm.py --cuda \
        --primitive_name push \
        --model_path 'joint_pushing_init_centered_2' \
        --model_number 65000 \
        --gnn_library $GNN_LIB
else
    echo 'Primitive not recognized, exiting'
fi        
