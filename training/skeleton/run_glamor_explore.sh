source ~/environments/py36-gnn/bin/activate
#if [ -z $1 ]
#then
#echo 'Pretraining!'
#python train_glamor_explore.py \
#	--cuda \
#	--batch_size 4 \
#	--pretrain \
#	--num_pretrain_epoch 10 \
#	--exp glamor_explore_scene_context_test
#else
#echo 'Resuming!'
#python train_glamor_explore.py \
#	--cuda \
#	--batch_size 4 \
#	--exp glamor_explore_scene_context_test \
#	--resume_iter 1000
#fi 
python train_glamor_explore.py \
	--cuda \
	--batch_size 4 \
	--exp glamor_explore_no_pretrain_table_only \
        --resume_iter 29000
