from planning import pushing_planning, grasp_planning, levering_planning, pulling_planning
from helper import util, planning_helper, collisions

import os
# from example_config import get_cfg_defaults
from closed_loop_experiments import get_cfg_defaults

from airobot import Robot
from airobot.utils import pb_util, common, arm_util
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from yacs.config import CfgNode as CN
import pybullet as p
import time
import argparse
import numpy as np
import threading
from multiprocessing import Process, Pipe, Queue

import pickle
import rospy
from IPython import embed

import trimesh
import copy

from macro_actions import ClosedLoopMacroActions, YumiGelslimPybulet
import signal
import open3d


def main(args):
    cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    rospy.init_node('replay')

    with open(os.join(args.data_dir, args.data_file), 'rb') as data_f:
        data = pickle.load(data_f)

    yumi_ar = Robot('yumi',
                    pb=True,
                    arm_cfg={'render': True,
                             'self_collision': False})

    if args.object:
        box_id = pb_util.load_urdf(
            args.config_package_path +
            'descriptions/urdf/'+args.object_name+'.urdf',
            cfg.OBJECT_POSE_3[0:3],
            cfg.OBJECT_POSE_3[3:]
        )        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_package_path',
        type=str,
        default='/root/catkin_ws/src/config/')

    parser.add_argument(
        '--example_config_path',
        type=str,
        default='config')

    parser.add_argument(
        '--primitive',
        type=str,
        default='pull',
        help='which primitive to plan')

    parser.add_argument(
        '--simulate',
        type=int,
        default=1)

    parser.add_argument(
        '-o',
        '--object',
        action='store_true')

    parser.add_argument(
        '-re',
        '--replan',
        action='store_true',
        default=False)

    parser.add_argument(
        '--object_name',
        type=str,
        default='realsense_box')

    parser.add_argument(
        '-ex',
        '--execute_thread',
        action='store_true'
    )

    parser.add_argument(
        '--debug', action='store_true'
    )

    parser.add_argument(
        '-r', '--rolling',
        type=float, default=0.0,
        help='rolling friction value for pybullet sim'
    )

    parser.add_argument(
        '-v', '--visualize',
        action='store_true'
    )

    args = parser.parse_args()
    main(args)