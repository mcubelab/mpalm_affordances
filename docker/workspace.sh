export CODE_BASE="$HOME"
cd $HOME/catkin_ws
catkin build config pykdl_utils abb_robotnode realsense2_camera hand_eye_calibration
# pip uninstall pyassimp -y
# pip install pyassimp==4.1.3
