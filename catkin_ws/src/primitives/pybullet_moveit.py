from planning import pushing_planning, grasp_planning, levering_planning, pulling_planning
from helper import util, planning_helper, collisions

import os
from example_config import get_cfg_defaults

from airobot import Robot
from airobot.utils import pb_util
import pybullet as p
import time
import argparse
import numpy as np

import pickle

# from motion_planning.motion_planning import MotionPlanner
from motion_planning.group_planner import GroupPlanner

import numpy as np
import copy
import rospy
import trajectory_msgs
import moveit_commander
import moveit_msgs
from moveit_commander.exception import MoveItCommanderException
from moveit_msgs.srv import GetStateValidity
# from ik import ik_helper

from IPython import embed
from scipy.interpolate import UnivariateSpline


data = {}
data['palm_pose_world'] = []
data['object_pose_palm'] = []
data['contact_bool'] = []

def get_tip_to_wrist(tip_poses, cfg):
    tip_to_wrist = util.list2pose_stamped(cfg.TIP_TO_WRIST_TF, '')
    world_to_world = util.unit_pose()

    wrist_left = util.convert_reference_frame(
        tip_to_wrist,
        world_to_world,
        tip_poses[0],
        "yumi_body")
    wrist_right = util.convert_reference_frame(
        tip_to_wrist,
        world_to_world,
        tip_poses[1],
        "yumi_body")

    # wrist_left = util.pose_stamped2list(wrist_left)
    # wrist_right = util.pose_stamped2list(wrist_right)

    return wrist_right.pose, wrist_left.pose

def get_joint_poses(tip_poses, robot, cfg, nullspace=True):
    tip_to_wrist = util.list2pose_stamped(cfg.TIP_TO_WRIST_TF, '')
    world_to_world = util.unit_pose()

    r_joints, l_joints = None, None

    wrist_left = util.convert_reference_frame(
        tip_to_wrist,
        world_to_world,
        tip_poses[0],
        "yumi_body")
    wrist_right = util.convert_reference_frame(
        tip_to_wrist,
        world_to_world,
        tip_poses[1],
        "yumi_body")

    wrist_left = util.pose_stamped2list(wrist_left)
    wrist_right = util.pose_stamped2list(wrist_right)

    # r_joints = robot._accurate_ik(
    # 	wrist_right[0:3],
    # 	wrist_right[3:],
    # 	arm='right',
    # 	nullspace=nullspace)[:7]

    # l_joints = robot._accurate_ik(
    # 	wrist_left[0:3],
    # 	wrist_left[3:],
    # 	arm='left',
    # 	nullspace=nullspace)[7:]

    #     wrist_right[3:])

    # print("r_joints:")
    r_joints = robot.arm.compute_ik(
        wrist_right[0:3],
        wrist_right[3:],
        arm='right',
        ns=nullspace)

    # print("l_joints:")
    l_joints = robot.arm.compute_ik(
        wrist_left[0:3],
        wrist_left[3:],
        arm='left',
        ns=nullspace)

    # l_joints = robot.arm.left_arm.compute_ik(
    #     wrist_left[0:3],
    #     wrist_left[3:])

    # r_joints = robot.arm.right_arm.compute_ik(
    #     wrist_right[0:3],


    return r_joints, l_joints, wrist_right, wrist_left


def align_arms(r_points, l_points):
    largest = max(len(r_points), len(l_points))
    r_points_mat = np.zeros((len(r_points), 7))
    l_points_mat = np.zeros((len(l_points), 7))

    for i, point in enumerate(r_points):
        r_points_mat[i, :] = point.positions
    for i, point in enumerate(l_points):
        l_points_mat[i, :] = point.positions

    new_r_points = np.zeros((largest, 7))
    new_l_points = np.zeros((largest, 7))

    if len(r_points) < largest:
        for i in range(7):
            old_jnt = r_points_mat[:, i]
            old_inds = np.arange(0, len(old_jnt))
            
            spl = UnivariateSpline(old_inds, old_jnt, k=3, s=0)

            new_jnt = spl(np.linspace(0, len(old_jnt)-1, largest))
            new_r_points[:, i] = copy.deepcopy(new_jnt)
        new_l_points = copy.deepcopy(l_points_mat)
    elif len(l_points) < largest:
        for i in range(7):
            old_jnt = l_points_mat[:, i]
            old_inds = np.arange(0, len(old_jnt))

            spl = UnivariateSpline(old_inds, old_jnt, k=3, s=0)

            new_jnt = spl(np.linspace(0, len(old_jnt)-1, largest))
            new_l_points[:, i] = copy.deepcopy(new_jnt)
        new_r_points = copy.deepcopy(r_points_mat)
    return new_r_points, new_l_points


def main(args):
    print(args)
    rospy.init_node('test')

    object_urdf = args.config_package_path+'descriptions/urdf/'+args.object_name+'.urdf'
    object_mesh = args.config_package_path+'descriptions/meshes/objects'+args.object_name+'.stl'

    moveit_robot = moveit_commander.RobotCommander()
    moveit_scene = moveit_commander.PlanningSceneInterface()
    moveit_planner = 'RRTconnectkConfigDefault'
    # moveit_planner = 'RRTstarkConfigDefault'

    mp_left = GroupPlanner(
        'left_arm',
        moveit_robot,
        moveit_planner,
        moveit_scene,
        max_attempts=50,
        planning_time=5.0,
        goal_tol=0.5,
        eef_delta=0.01, 
        jump_thresh=10.0)

    mp_right = GroupPlanner(
        'right_arm',
        moveit_robot,
        moveit_planner,
        moveit_scene,
        max_attempts=50,
        planning_time=5.0,
        goal_tol=0.5,
        eef_delta=0.01,
        jump_thresh=10.0)

    cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()
    print(cfg)

    manipulated_object = None
    object_pose1_world = util.list2pose_stamped(cfg.OBJECT_INIT)
    object_pose2_world = util.list2pose_stamped(cfg.OBJECT_FINAL)
    palm_pose_l_object = util.list2pose_stamped(cfg.PALM_LEFT)
    palm_pose_r_object = util.list2pose_stamped(cfg.PALM_RIGHT)

    if args.primitive == 'push':
        plan = pushing_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object)

    elif args.primitive == 'grasp':
        plan = grasp_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object)

    elif args.primitive == 'pivot':
        gripper_name = args.config_package_path + \
            'descriptions/meshes/mpalm/mpalms_all_coarse.stl'
        table_name = args.config_package_path + \
            'descriptions/meshes/table/table_top.stl'

        manipulated_object = collisions.CollisionBody(
            args.config_package_path + 'descriptions/meshes/objects/realsense_box_experiments.stl')

        plan = levering_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            gripper_name=gripper_name,
            table_name=table_name)

    elif args.primitive == 'pull':
        plan = pulling_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            arm='r')

    else:
        raise NotImplementedError

    object_loaded = False
    box_id = None

    # from IPython import embed
    # embed()

    yumi = Robot('yumi',
                pb=True,
                arm_cfg={'render': True, 'self_collision': False})
    # yumi.arm.go_home()
    yumi.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    # embed() 

    box_id = pb_util.load_urdf(
        args.config_package_path+'descriptions/urdf/'+args.object_name+'.urdf',
        cfg.OBJECT_INIT[0:3],
        cfg.OBJECT_INIT[3:]
    )

    last_tip_right = None
    for plan_dict in plan:

        tip_poses = plan_dict['palm_poses_world']
        
        wrist_right = []
        wrist_left = []

        tip_right = []
        tip_left = []

        for i in range(len(tip_poses)):
            tip_left.append(tip_poses[i][0].pose)
            tip_right.append(tip_poses[i][1].pose)

        # tip_right.append(util.pose_stamped2list(tip_poses[0][1]))
        # tip_right.append(util.pose_stamped2list(tip_poses[-1][1]))

        # tip_right.append(tip_poses[0][1].pose)
        # tip_right.append(tip_poses[-1][1].pose)
        
        l_current = yumi.arm.get_jpos()[7:]
        r_current = yumi.arm.get_jpos()[:7]

        l_plan = mp_left.plan_waypoints(
            tip_left, 
            force_start=l_current+r_current, 
            avoid_collisions=False)
        r_plan = mp_right.plan_waypoints(
            tip_right, 
            force_start=l_current+r_current, 
            avoid_collisions=False)

        sleep_t = 0.005
        loop_t = 0.125

        sync_r_plan, sync_l_plan = align_arms(r_plan.points, l_plan.points)

        # embed()

        # for i in range(len(r_plan.points)):
        #     pos = r_plan.points[i].positions
        #     start = time.time()
        #     while time.time() - start < loop_t:
        #         yumi.arm.set_jpos(pos, arm='right')
        #         time.sleep(sleep_t)

        # for i in range(len(l_plan.points)):
        #     pos = l_plan.points[i].positions
        #     start = time.time()
        #     while time.time() - start < loop_t:
        #         yumi.arm.set_jpos(pos, arm='left', wait=False)
        #         time.sleep(sleep_t)        

        for i in range(sync_r_plan.shape[0]):
            r_pos = sync_r_plan[i, :]
            l_pos = sync_l_plan[i, :]

            start = time.time()
            while time.time() - start < loop_t:
                yumi.arm.set_jpos(np.hstack((r_pos,l_pos)))
                # yumi.arm.set_jpos(r_pos, arm='right')
                time.sleep(sleep_t)

        # embed()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_package_path',
                        type=str,
                        default='/home/anthony/repos/research/mpalm_affordances/catkin_ws/src/config/')
    parser.add_argument('--example_config_path', type=str, default='config')
    parser.add_argument('--primitive', type=str, default='push', help='which primitive to plan')
    parser.add_argument('--simulate', type=int, default=1)
    parser.add_argument('--object', type=int, default=0)
    parser.add_argument('--object_name', type=str, default='realsense_box')
    args = parser.parse_args()
    main(args)
