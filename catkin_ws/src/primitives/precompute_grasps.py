import os
import os.path as osp
import sys
import time
import argparse
import numpy as np
import rospy
import signal
import threading
from multiprocessing import Pipe, Queue
import pickle
import open3d
import copy
from random import shuffle
from IPython import embed

from airobot import Robot
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions
from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

from multistep_planning_eval_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler
from data_gen_utils import YumiCamsGS, DataManager, MultiBlockManager, GoalVisual
import simulation
from helper import registration as reg
from eval_utils.visualization_tools import PCDVis, PalmVis
from eval_utils.experiment_recorder import GraspEvalManager
from helper.pointcloud_planning import (
    PointCloudNode, 
    GraspSamplerVAEPubSub, PullSamplerVAEPubSub,
    GraspSamplerTransVAEPubSub,
    GraspSamplerBasic, PullSamplerBasic)
from planning import grasp_planning_wf, pulling_planning_wf

import task_planning.sampling as sampling
import task_planning.grasp_sampling as grasp_sampling
import task_planning.lever_sampling as lever_sampling
from task_planning.objects import Object, CollisionBody
import tf
import random

from helper import util

import os
# from example_config import get_cfg_defaults
from closed_loop_experiments_cfg import get_cfg_defaults

from airobot import Robot
# from airobot.utils import pb_util, common
from airobot.utils import common
import pybullet as p
import time
import argparse
import numpy as np
import threading

import pickle
import rospy
from IPython import embed

import trimesh
import copy

from macro_actions import ClosedLoopMacroActions  # YumiGelslimPybulet
from yumi_pybullet_ros import YumiGelslimPybullet

def main(args):
    cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    rospy.init_node('precompute')
    np_seed = args.np_seed
    np.random.seed(np_seed)

    # setup yumi
    yumi_ar = Robot('yumi_palms',
                    pb=True,
                    pb_cfg={'gui': False, 'realtime': True},
                    arm_cfg={'self_collision': False, 'seed': np_seed})
    yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT, ignore_physics=True)

    yumi_gs = YumiCamsGS(
        yumi_ar,
        cfg,
        exec_thread=True)

    # yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    ycb_object_files = os.listdir(os.path.join(args.config_package_path, 'descriptions/meshes/objects/ycb_objects'))
    cylinder_object_files = os.listdir(os.path.join(args.config_package_path, 'descriptions/meshes/objects/cylinders'))
    

    for object_name in ycb_object_files:
    # for object_name in cylinder_object_files:
        # if object_name in ['banana_down.stl', 'bowl_down.stl', 'windex_down.stl', 'banana_down.stl', 'bowl_down.stl', 'lemon_down.stl', 'master_down.stl', 'mustard_1k.stl', 'potted_down.stl', 'pudding_down.stl', 'original']:
        #     continue 
        if 'power_down' not in object_name:
            continue
        stl_file = os.path.join(args.config_package_path, 'descriptions/meshes/objects/ycb_objects', object_name)
        # stl_file = os.path.join(args.config_package_path, 'descriptions/meshes/objects/cylinders', object_name)
        print(stl_file)
        tmesh = trimesh.load_mesh(stl_file)
        init_pose = tmesh.compute_stable_poses()[0][0]
        pos = init_pose[:-1, -1]
        ori = common.rot2quat(init_pose[:-1, :-1])
        box_id = yumi_ar.pb_client.load_geom(
            shape_type='mesh', 
            visualfile=stl_file, 
            collifile=stl_file, 
            mesh_scale=[1.0, 1.0, 1.0],
            base_pos=[0.45, 0, pos[-1]],
            base_ori=ori, 
            rgba=[0.7, 0.2, 0.2, 1.0],
            mass=0.03)      

        primitive_name = 'grasp'

        # mesh_file = args.config_package_path + \
        #             'descriptions/meshes/objects/' + \
        #             args.object_name + '.stl'
        # mesh_file = args.config_package_path + \
        #             'descriptions/meshes/objects/cylinders/' + object_name    
        mesh_file = args.config_package_path + \
                    'descriptions/meshes/objects/ycb_objects/' + object_name    

        if not os.path.exists(os.path.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives/data/grasp_samples', object_name.split('.stl')[0])):
            os.makedirs(os.path.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives/data/grasp_samples', object_name.split('.stl')[0]))

        exp_double = DualArmPrimitives(
            cfg,
            yumi_ar.pb_client.get_client_id(),
            box_id,
            mesh_file)

        coll_samples_file = os.path.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives/data/grasp_samples', object_name.split('.stl')[0], 'collision_free_samples.pkl')
        print('saving to: ')
        print(coll_samples_file)
        with open(coll_samples_file, 'wb') as f:
            pickle.dump(exp_double.grasp_samples.collision_free_samples, f)

        yumi_ar.pb_client.remove_body(box_id)

        # embed()


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
        '--np_seed', type=int,
        default=0
    )

    parser.add_argument(
        '--perturb',
        action='store_true'
    )

    args = parser.parse_args()
    main(args)
