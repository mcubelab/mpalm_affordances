export CONTAINER_ID=$(docker ps -l -q)
echo "Copying to ${CONTAINER_ID}"

# copy to known airobot locations
docker cp $PWD/yumi_gelslim_palm_shelf.urdf $CONTAINER_ID:/home/anthony/airobot/src/airobot/urdfs
docker cp $PWD/shelf_back.stl $CONTAINER_ID:/home/anthony/airobot/src/airobot/urdfs/meshes/objects
docker cp $PWD/rgbdcam.py $CONTAINER_ID:/home/anthony/airobot/src/airobot/sensor/camera

