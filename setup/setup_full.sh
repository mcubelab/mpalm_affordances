# make sure rpo_planning is built in the workspace
SETUP_DIR=${PWD}
cd ~/catkin_ws/src
catkin build rpo_planning
source devel/setup.bash

# make sure LCM is built inside the python 3 virtualenv
cd ${SETUP_DIR} 
source ~/environments/py36-gnn/bin/activate
python setup_lcm.py

cd ~/airobot && pip install -e .

deactivate

# download all model weights and pretraining data
cd ~/training/gat/data_utils; ./run_dl_all_weights.sh
cd ~/training/skeleton/data_utils; ./run_dl_all.sh

# download rearrangement training tasks
cd ~/catkin_ws/src/rpo_planning/src/rpo_planning/exploration; ./run_dl_training_tasks.sh
