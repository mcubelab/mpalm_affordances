#if [ $1 = "-g" ]
#then
#    IMAGE=anthonysimeonov/mpalm-dev-gpu:0.1.2
#    RUN_ARGS="--net=host --runtime=nvidia"
#elif [ $1 = "-p" ]
#then
#    IMAGE=anthonysimeonov/mpalm-dev-pytorch:0.1.2
#    RUN_ARGS="--net=host --runtime=nvidia"    
#else
#    IMAGE=anthonysimeonov/mpalm-dev-cpu:0.1.2
#    RUN_ARGS=""
#fi

#IMAGE=mpalm-dev-cpu
#IMAGE=mpalm-dev-gpu
IMAGE=anthonysimeonov/mpalm-dev-gpu:0.2.2
a=0
CONTAINER_NAME="$1-$a"
while [ $a -lt $2 ]
do
	CONTAINER_NAME="$1-$a"

	sudo docker run -d \
	    --name=$CONTAINER_NAME \
	    --volume="$PWD/../catkin_ws/src/:/home/asimeono/catkin_ws/src/" \
	    --volume="$PWD/workspace.sh:/workspace.sh" \
	    --volume="$PWD/../training/:/home/asimeono/training/" \
	    --volume="$PWD/../../airobot/:/home/asimeono/airobot/" \
	    --user="asimeono" \
	    ${IMAGE}
	#echo $a
	echo $CONTAINER_NAME
	a=`expr $a + 1`
done

