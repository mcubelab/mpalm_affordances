source ~/environments/py36-gnn/bin/activate
if [ -z $1 ]
then
echo 'Pretraining!'
python train_glamor_explore.py \
	--cuda \
	--batch_size 8 \
	--pretrain \
	--num_pretrain_epoch 10 \
	--exp glamor_explore_test
else
echo 'Resuming!'
python train_glamor_explore.py \
	--cuda \
	--batch_size 8 \
	--exp glamor_explore_test \
	--resume_iter 1000
fi 
