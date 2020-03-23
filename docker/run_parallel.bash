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

#IMAGE=anthonysimeonov/mpalm-dev-cpu:0.1.2

#XAUTH=/tmp/.docker.xauth
#if [ ! -f $XAUTH ]
#then
#    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
#    if [ ! -z "$xauth_list" ]
#    then
#        echo $xauth_list | xauth -f $XAUTH nmerge -
#    else
#        touch $XAUTH
#    fi
#    chmod a+r $XAUTH
#fi

#IMAGE=mpalm-dev-cpu
IMAGE=mpalm-dev-gpu
a=0
CONTAINER_NAME="data-gen-$a"
while [ $a -lt $1 ]
do
	CONTAINER_NAME="data-gen-$a"

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
#sudo docker run -d \
#    --volume="$PWD/../catkin_ws/src/:/home/asimeono/catkin_ws/src/" \
#    --volume="$PWD/workspace.sh:/workspace.sh" \
#    --volume="$PWD/../training/:/home/asimeono/training/" \
#    --volume="$PWD/../../airobot/:/home/asimeono/airobot/" \
#    --user="asimeono" \
#    ${IMAGE}
