import os, sys
sys.path.append('../catkin_ws/src/primitives/')
import pickle
import matplotlib.pyplot
import numpy as np
from mpl_toolkits.mplot3d import axes3d

# os.system('export CODE_BASE=/root/')

import time
import argparse
import numpy as np
from multiprocessing import Process, Pipe, Queue
import pickle
import rospy
import copy
import signal
import open3d
from IPython import embed

from airobot import Robot
from airobot.utils import pb_util
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions, YumiGelslimPybulet
# from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

from yacs.config import CfgNode as CN
from closed_loop_experiments import get_cfg_defaults

with open('data/pull/face_ind_large_0/metadata.pkl', 'rb') as mf:
    metadata = pickle.load(mf)

print('Metadata keys: ')
print(metadata.keys())
print(metadata['cfg'])
cfg = metadata['cfg']
step_repeat = metadata['step_repeat']
dynamics_info = metadata['dynamics']
mesh_file = metadata['mesh_file']

with open ('data/pull/face_ind_large_0/1.pkl', 'rb') as f:
    data = pickle.load(f)

    
rospy.init_node("test")
yumi = Robot('yumi_palms', arm_cfg={'render': True, 'self_collision': False, 'rt_simulation': True})
yumi.arm.go_home()
yumi.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT)

gel_id = 12

p.changeDynamics(
    yumi.arm.robot_id,
    gel_id,
    restitution=dynamics_info['restitution'],
    contactStiffness=dynamics_info['contactStiffness'],
    contactDamping=dynamics_info['contactDamping'],
    rollingFriction=dynamics_info['rollingFriction']
)

yumi_gs = YumiGelslimPybulet(
    yumi,
    cfg)

box_id = pb_util.load_urdf(
    '/root/catkin_ws/src/config/descriptions/urdf/realsense_box.urdf',
    cfg.OBJECT_POSE_3[0:3],
    cfg.OBJECT_POSE_3[3:]
)

config_pkg_path = '/root/catkin_ws/src/config/'

action_planner = ClosedLoopMacroActions(
    cfg,
    yumi_gs,
    box_id,
    pb_util.PB_CLIENT,
    config_pkg_path,
    object_mesh_file=mesh_file,
    replan=True
)

from helper.util import pose_stamped2list, list2pose_stamped

planner_args = data['planner_args']
object_start_pose_list = data['start']
object_goal_pose_list = data['goal']

print(planner_args.keys())
print(planner_args['primitive_name'])
primitive_name = planner_args['primitive_name']

pb_util.reset_body(
    body_id=box_id, 
    base_pos=object_start_pose_list[:3],
    base_quat=object_start_pose_list[3:])

new_args = {}
new_args['object_pose1_world'] = list2pose_stamped(object_start_pose_list)
new_args['object_pose2_world'] = list2pose_stamped(object_goal_pose_list)
new_args['primitive_name'] = 'pull'
new_args['palm_pose_r_object'] = list2pose_stamped(data['contact_obj_frame'])
new_args['palm_pose_l_object'] = list2pose_stamped(cfg.PALM_LEFT)
new_args['object'] = None
new_args['init'] = True
new_args['N'] = 32
new_args['table_face'] = 0

from planning import pulling_planning

manipulated_object = new_args['object']
object_pose1_world = new_args['object_pose1_world']
object_pose2_world = new_args['object_pose2_world']
palm_pose_l_object = new_args['palm_pose_l_object']
palm_pose_r_object = new_args['palm_pose_r_object']
table_face = new_args['table_face']
active_arm = 'right'
N = new_args['N']

plan = pulling_planning(
    object=manipulated_object,
    object_pose1_world=object_pose1_world,
    object_pose2_world=object_pose2_world,
    palm_pose_l_object=palm_pose_l_object,
    palm_pose_r_object=palm_pose_r_object,
    arm=active_arm[0],
    N=N)

# from IPython import embed
# embed()

# valid = action_planner.full_mp_check(plan, 'pull')
result = action_planner.execute(primitive_name=primitive_name, execute_args=planner_args)

from IPython import embed
embed()


