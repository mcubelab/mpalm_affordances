rosrun xacro xacro --inorder -o yumi_gelslim_palm.urdf yumi_setup.urdf.xacro name:=yumi gripper:=yumi_gelslim_palm
urdf_to_graphiz yumi_gelslim_palm.urdf
rm yumi_gelslim_palm.gv