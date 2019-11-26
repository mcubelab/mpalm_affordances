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
from motion_planning.motion_planning import MotionPlanner

import numpy as np
import copy
import rospy
import trajectory_msgs
import moveit_commander
import moveit_msgs
from moveit_commander.exception import MoveItCommanderException
from moveit_msgs.srv import GetStateValidity
# from ik import ik_helper
from tactile_helper.ik import ik_helper

from IPython import embed
from scipy.interpolate import UnivariateSpline


import rospy
from trac_ik_python import trac_ik

data = {}
data['palm_pose_world'] = []
data['object_pose_palm'] = []
data['contact_bool'] = []


def get_active_arm(object_init_pose):
    """
    Returns whether the right arm or left arm
    should be used for pushing, depending on which
    arm the object is closest to

    Args:
        object_init_pose (list): Initial pose of the
            object on the table, in form [x, y, z, x, y, z, w]

    Returns:
        str: 'right' or 'left'
    """
    if object_init_pose[1] > 0:
        active_arm = 'left'
    else:
        active_arm = 'right'

    return active_arm


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


def unify_arm_trajectories(left_arm, right_arm, tip_poses):
    """
    Function to return a right arm and left arm trajectory
    of the same number of points, where the index of the points
    that align with the goal cartesian poses of each arm are the
    same for both trajectories

    Args:
        left_arm (JointTrajectory): left arm joint trajectory returned
            by left arm move group after calling compute_cartesian_path
        right_arm (JointTrajectory): right arm joint trajectory returned
            by right arm move group after calling compute_cartesian_path
        tip_poses (list): list of desired end effector poses to follow for
            both arms for a particular segment of a primitive plan

    Returns:

    """
    # find the longer trajectory
    long_traj = 'left' if len(left_arm.points) > len(right_arm.points) else 'right'

    # make numpy array of each arm joint trajectory for each comp
    left_arm_joints_np = np.zeros((len(left_arm.points), 7))
    right_arm_joints_np = np.zeros((len(right_arm.points), 7))

    # make numpy array of each arm pose trajectory, based on fk
    left_arm_fk_np = np.zeros((len(left_arm.points), 7))
    right_arm_fk_np = np.zeros((len(right_arm.points), 7))

    for i, point in enumerate(left_arm.points):
        left_arm_joints_np[i, :] = point.positions
        pose = ik_helper.compute_fk(point.positions, arm='l')
        left_arm_fk_np[i, :] = util.pose_stamped2list(pose)
    for i, point in enumerate(right_arm.points):
        right_arm_joints_np[i, :] = point.positions
        pose = ik_helper.compute_fk(point.positions, arm='r')
        right_arm_fk_np[i, :] = util.pose_stamped2list(pose)

    closest_left_inds = []
    closest_right_inds = []

    # for each tip_pose, find the index in the longer trajectory that
    # most closely matches the pose (using fk)
    for i in range(len(tip_poses)):
        r_waypoint = util.pose_stamped2list(tip_poses[i][1])
        l_waypoint = util.pose_stamped2list(tip_poses[i][0])

        r_pos_diffs = util.pose_difference_np(
            pose=right_arm_fk_np, pose_ref=np.array(r_waypoint))[0]
        l_pos_diffs = util.pose_difference_np(
            pose=left_arm_fk_np, pose_ref=np.array(l_waypoint))[0]

        r_index = np.argmin(r_pos_diffs)
        l_index = np.argmin(l_pos_diffs)

        closest_right_inds.append(r_index)
        closest_left_inds.append(l_index)

    # Create a new trajectory for the shorter trajectory, that is the same
    # length as the longer trajectory.

    if long_traj == 'l':
        new_right = np.zeros((left_arm_joints_np.shape))
        prev_r_ind = 0
        prev_new_ind = 0

        for i, r_ind in enumerate(closest_right_inds):
            # Put the joint values from the short
            # trajectory at the indices corresponding to the path waypoints
            # at the corresponding indices found for the longer trajectory
            new_ind = closest_left_inds[i]
            new_right[new_ind, :] = right_arm_joints_np[r_ind, :]

            # For the missing values in between the joint waypoints,
            # interpolate to fill the trajectory
            interp = np.linspace(
                right_arm_joints_np[prev_r_ind, :],
                right_arm_joints_np[r_ind],
                num=new_ind - prev_new_ind)

            new_right[prev_new_ind:new_ind, :] = interp

            prev_r_ind = r_ind
            prev_new_ind = new_ind

        aligned_right_joints = new_right
        aligned_left_joints = left_arm_joints_np
    else:
        new_left = np.zeros((right_arm_joints_np.shape))
        prev_l_ind = 0
        prev_new_ind = 0

        for i, l_ind in enumerate(closest_left_inds):
            new_ind = closest_right_inds[i]
            new_left[new_ind, :] = left_arm_joints_np[l_ind, :]

            interp = np.linspace(
                left_arm_joints_np[prev_l_ind, :],
                left_arm_joints_np[l_ind],
                num=new_ind - prev_new_ind)

            new_left[prev_new_ind:new_ind, :] = interp

            prev_l_ind = l_ind
            prev_new_ind = new_ind

        aligned_right_joints = right_arm_joints_np
        aligned_left_joints = new_left

    unified = {}
    unified['right'] = {}
    unified['right']['fk'] = right_arm_fk_np
    unified['right']['joints'] = right_arm_joints_np
    unified['right']['aligned_joints'] = aligned_right_joints
    unified['right']['inds'] = closest_right_inds

    unified['left'] = {}
    unified['left']['fk'] = left_arm_fk_np
    unified['left']['joints'] = left_arm_joints_np
    unified['left']['aligned_joints'] = aligned_left_joints
    unified['left']['inds'] = closest_left_inds
    return unified

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

    active_arm = get_active_arm(cfg.OBJECT_INIT)

    if args.primitive == 'push':
        plan = pushing_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            arm=active_arm[0])

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
            arm=active_arm[0])

    else:
        raise NotImplementedError

    object_loaded = False
    box_id = None

    yumi = Robot('yumi',
                pb=True,
                arm_cfg={'render': True, 'self_collision': False})
    # yumi.arm.go_home()
    yumi.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    if args.object:
        box_id = pb_util.load_urdf(
            args.config_package_path+'descriptions/urdf/'+args.object_name+'.urdf',
            cfg.OBJECT_INIT[0:3],
            cfg.OBJECT_INIT[3:]
        )

    sleep_t = 0.005
    loop_t = 0.125

    for plan_dict in plan:

        tip_poses = plan_dict['palm_poses_world']

        tip_right = []
        tip_left = []

        for i in range(len(tip_poses)):
            tip_left.append(tip_poses[i][0].pose)
            tip_right.append(tip_poses[i][1].pose)

        l_current = yumi.arm.get_jpos()[7:]
        r_current = yumi.arm.get_jpos()[:7]

        if args.primitive == 'pivot' or args.primitive == 'grasp':
            traj_right = mp_right.plan_waypoints(
                tip_right,
                force_start=l_current+r_current,
                avoid_collisions=False)

            traj_left = mp_left.plan_waypoints(
                tip_left,
                force_start=l_current+r_current,
                avoid_collisions=False)

            unified = unify_arm_trajectories(
                traj_left,
                traj_right,
                tip_poses)

            aligned_left = unified['left']['aligned_joints']
            aligned_right = unified['right']['aligned_joints']

            if aligned_left.shape != aligned_right.shape:
                raise ValueError('Could not aligned joint trajectories')
                return

            for i in range(aligned_right.shape[0]):
                r_pos = aligned_right[i, :]
                l_pos = aligned_left[i, :]

                start = time.time()
                while time.time() - start < loop_t:
                    yumi.arm.set_jpos(np.hstack((r_pos, l_pos)), wait=False)
                    time.sleep(sleep_t)
        else:
            if active_arm == 'right':
                traj = mp_right.plan_waypoints(
                    tip_right,
                    force_start=l_current+r_current,
                    avoid_collisions=False)
            else:
                traj = mp_left.plan_waypoints(
                    tip_left,
                    force_start=l_current+r_current,
                    avoid_collisions=False)

            for i, point in enumerate(traj.points):
                j_pos = point.positions
                start = time.time()

                while time.time() - start < loop_t:
                    yumi.arm.set_jpos(j_pos, arm=active_arm, wait=False)
                    time.sleep(sleep_t)


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
