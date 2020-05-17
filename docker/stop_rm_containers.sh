CONTAINER_NAME=$1
echo "Stopping containers matching name: $1"
for i in 0 1 2 3 4 5 6 7 8 9;do sudo docker stop "$1-$i"; done
echo "Removing containers matching name: $1"
for i in 0 1 2 3 4 5 6 7 8 9;do sudo docker rm "$1-$i"; done
