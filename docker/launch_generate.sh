#!/bin/bash

base_name="$1"
echo "Launching data gen in containers matching name: $base_name"
for a in `sudo docker ps --filter "name=$base_name" --format "{{.Names}}"`
do
	echo $a
	sudo docker exec -d -w /home/asimeono/catkin_ws/src/primitives $a ./generate.sh
	sleep 5
done
