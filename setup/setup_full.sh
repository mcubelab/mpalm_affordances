# make sure rpo_planning is built in the workspace
SETUP_DIR=${PWD}
cd ~/catkin_ws/src
catkin build rpo_planning

# make sure LCM is built inside the python 3 virtualenv
cd ${SETUP_DIR} 
source ~/environments/py36-gnn/bin/activate
python setup_lcm.py

cd ~/airobot && pip install -e .

deactivate
