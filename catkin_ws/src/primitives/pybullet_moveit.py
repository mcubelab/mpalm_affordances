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
import threading

data = {}
data['palm_pose_world'] = []
data['object_pose_palm'] = []
data['contact_bool'] = []

global joint_lock

class IKHelper():
    def __init__(self):
        robot_description = '/robot_description'
        urdf_string = rospy.get_param(robot_description)
        self.num_ik_solver_r = trac_ik.IK('yumi_body', 'yumi_tip_r',
                                          urdf_string=urdf_string)

        self.num_ik_solver_l = trac_ik.IK('yumi_body', 'yumi_tip_l',
                                          urdf_string=urdf_string)

    def compute_ik(self, pos, ori, seed, arm='right'):
        if arm != 'right' and arm != 'left':
            arm = 'right'
        if arm == 'right':
            sol = self.num_ik_solver_r.get_ik(
                seed,
                pos[0],
                pos[1],
                pos[2],
                ori[0],
                ori[1],
                ori[2],
                ori[3],
                0.01, 0.01, 0.01,
                0.1, 0.1, 0.1
            )
        elif arm == 'left':
            sol = self.num_ik_solver_l.get_ik(
                seed,
                pos[0],
                pos[1],
                pos[2],
                ori[0],
                ori[1],
                ori[2],
                ori[3],
                0.01, 0.01, 0.01,
                0.1, 0.1, 0.1
            )
        return sol


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
        unactive_arm = 'right'
    else:
        active_arm = 'right'
        unactive_arm = 'left'

    return active_arm, unactive_arm


def get_tip_to_wrist(tip_poses, cfg):
    """
    [summary]

    Args:
        tip_poses ([type]): [description]
        cfg ([type]): [description]

    Returns:
        [type]: [description]
    """
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

    return wrist_right.pose, wrist_left.pose


def get_joint_poses(tip_poses, robot, cfg, nullspace=True):
    """
    [summary]

    Args:
        tip_poses ([type]): [description]
        robot ([type]): [description]
        cfg ([type]): [description]
        nullspace (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
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

    r_joints = robot.arm.compute_ik(
        wrist_right[0:3],
        wrist_right[3:],
        arm='right',
        ns=nullspace)

    l_joints = robot.arm.compute_ik(
        wrist_left[0:3],
        wrist_left[3:],
        arm='left',
        ns=nullspace)
    return r_joints, l_joints, wrist_right, wrist_left


def align_arms(r_points, l_points):
    """
    [summary]

    Args:
        r_points ([type]): [description]
        l_points ([type]): [description]

    Returns:
        [type]: [description]
    """
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


def get_primitive_plan(primitive_name, primitive_args,
                       config_path, active_arm):
    """
    [summary]

    Args:
        primitive_name ([type]): [description]
        primitive_args ([type]): [description]
        config_path ([type]): [description]
        active_arm ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    manipulated_object = primitive_args['object']
    object_pose1_world = primitive_args['object_pose1_world']
    object_pose2_world = primitive_args['object_pose2_world']
    palm_pose_l_object = primitive_args['palm_pose_l_object']
    palm_pose_r_object = primitive_args['palm_pose_r_object']

    if primitive_name == 'push':
        plan = pushing_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            arm=active_arm[0])

    elif primitive_name == 'grasp':
        plan = grasp_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object)

    elif primitive_name == 'pivot':
        gripper_name = config_path + \
            'descriptions/meshes/mpalm/mpalms_all_coarse.stl'
        table_name = config_path + \
            'descriptions/meshes/table/table_top.stl'

        manipulated_object = collisions.CollisionBody(
            config_path +
            'descriptions/meshes/objects/realsense_box_experiments.stl')

        plan = levering_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            gripper_name=gripper_name,
            table_name=table_name)

    elif primitive_name == 'pull':
        plan = pulling_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            arm=active_arm[0])
    else:
        raise ValueError('Primitive name not recognized')

    return plan


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
    long_traj = 'left' if len(left_arm.points) > len(right_arm.points) \
        else 'right'

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


def greedy_replan(yumi, active_arm, box_id, primitive, object_final_pose,
                  config_path, ik, seed, iteration, plan_iteration=0):
    """
    [summary]
    """
    global initial_plan
    # get the current inputs to the planner
    # both palms in the object frame, current pose of the object, active arm
    object_pos = list(yumi.arm.p.getBasePositionAndOrientation(box_id)[0])
    object_ori = list(yumi.arm.p.getBasePositionAndOrientation(box_id)[1])

    r_tip_pos_world = list(yumi.arm.p.getLinkState(yumi.arm.robot_id, 13)[0])
    r_tip_ori_world = list(yumi.arm.p.getLinkState(yumi.arm.robot_id, 13)[1])

    l_tip_pos_world = list(yumi.arm.p.getLinkState(yumi.arm.robot_id, 26)[0])
    l_tip_ori_world = list(yumi.arm.p.getLinkState(yumi.arm.robot_id, 26)[1])

    r_tip_pose_object_frame = util.convert_reference_frame(
        util.list2pose_stamped(r_tip_pos_world + r_tip_ori_world),
        util.list2pose_stamped(object_pos + object_ori),
        util.unit_pose()
    )
    l_tip_pose_object_frame = util.convert_reference_frame(
        util.list2pose_stamped(l_tip_pos_world + l_tip_ori_world),
        util.list2pose_stamped(object_pos + object_ori),
        util.unit_pose()
    )

    object_pose_current = util.list2pose_stamped(object_pos + object_ori)

    primitive_args = {}
    primitive_args['object_pose1_world'] = object_pose_current
    primitive_args['object_pose2_world'] = object_final_pose
    primitive_args['palm_pose_l_object'] = l_tip_pose_object_frame
    primitive_args['palm_pose_r_object'] = r_tip_pose_object_frame
    primitive_args['object'] = None

    new_plan = get_primitive_plan(
        primitive, primitive_args, config_path, active_arm)

    new_tip_poses = new_plan[plan_iteration]['palm_poses_world'][0]

    seed_r = seed['right']
    seed_l = seed['left']

    print("old tip poses: ")
    print(util.pose_stamped2list(ik_helper.compute_fk(seed_r, arm='r')))
    print("new tip poses: ")
    print(util.pose_stamped2list(new_tip_poses[1]))
    print("new object pose: ")
    print(object_pos + object_ori)
    print("r_palm object frame: ")
    print(util.pose_stamped2list(r_tip_pose_object_frame))
    print("---")

    r_joints = ik.compute_ik(
        util.pose_stamped2list(new_tip_poses[1])[:3],
        util.pose_stamped2list(new_tip_poses[1])[3:],
        seed_r,
        arm='right'
    )

    l_joints = ik.compute_ik(
        util.pose_stamped2list(new_tip_poses[0])[:3],
        util.pose_stamped2list(new_tip_poses[0])[3:],
        seed_l,
        arm='left'
    )

    joints = {}
    joints['right'] = r_joints
    joints['left'] = l_joints

    # embed()

    return joints


def is_in_contact(yumi, box_id):
    r_pts = yumi.arm.p.getContactPoints(
        bodyA=yumi.arm.robot_id, bodyB=box_id, linkIndexA=12)
    l_pts = yumi.arm.p.getContactPoints(
        bodyA=yumi.arm.robot_id, bodyB=box_id, linkIndexA=25)
    
    contact_bool = 0 if (len(r_pts) == 0 and len(l_pts) == 0) else 1

    return contact_bool


def perturb_box(yumi, box_id, delta_pos, delta_ori=None):
    object_pos = list(yumi.arm.p.getBasePositionAndOrientation(box_id)[0])
    object_ori = list(yumi.arm.p.getBasePositionAndOrientation(box_id)[1])
    
    new_pos = np.array(object_pos) + np.array(delta_pos)
    if delta_ori is not None:
        pass
    new_ori = object_ori

    yumi.arm.p.resetBasePositionAndOrientation(
        box_id,
        new_pos.tolist(),
        new_ori
    )


def _yumi_execute_both(yumi):
    global joint_lock
    global both_pos

    while True:
        joint_lock.acquire()
        yumi.arm.set_jpos(both_pos, wait=False)
        joint_lock.release()
        time.sleep(0.005)


def _yumi_execute_arm(yumi, active_arm):
    global joint_lock
    global single_pos

    while True:
        joint_lock.acquire()
        yumi.arm.set_jpos(single_pos[active_arm], arm=active_arm, wait=False)
        joint_lock.release()
        time.sleep(0.005)


def main(args):
    global joint_lock
    joint_lock = threading.RLock()
    print(args)
    rospy.init_node('test')

    object_urdf = args.config_package_path + \
        'descriptions/urdf/'+args.object_name+'.urdf'
    object_mesh = args.config_package_path + \
        'descriptions/meshes/objects'+args.object_name+'.stl'

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

    active_arm, unactive_arm = get_active_arm(cfg.OBJECT_INIT)

    planner_args = {}
    planner_args['object_pose1_world'] = object_pose1_world
    planner_args['object_pose2_world'] = object_pose2_world
    planner_args['palm_pose_l_object'] = palm_pose_l_object
    planner_args['palm_pose_r_object'] = palm_pose_r_object
    planner_args['object'] = manipulated_object

    global initial_plan

    initial_plan = get_primitive_plan(
        args.primitive,
        planner_args,
        args.config_package_path,
        active_arm)

    box_id = None

    yumi = Robot('yumi',
                 pb=True,
                 arm_cfg={'render': True, 'self_collision': False})
    # yumi.arm.go_home()
    yumi.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    global both_pos
    global single_pos
    global joint_lock

    single_pos={}

    joint_lock.acquire()
    both_pos = cfg.RIGHT_INIT + cfg.LEFT_INIT
    print("both pos: ")
    print(both_pos)
    single_pos['right'] = cfg.RIGHT_INIT
    single_pos['left'] = cfg.LEFT_INIT
    joint_lock.release()

    if args.object:
        box_id = pb_util.load_urdf(
            args.config_package_path +
            'descriptions/urdf/'+args.object_name+'.urdf',
            cfg.OBJECT_INIT[0:3],
            cfg.OBJECT_INIT[3:]
        )
        # global trans_box_id
        # trans_box_id = pb_util.load_urdf(
        #     args.config_package_path + 
        #     'descriptions/urdf/'+args.object_name+'_trans.urdf',
        #     cfg.OBJECT_INIT[0:3],
        #     cfg.OBJECT_INIT[3:]
        # )

    sleep_t = 0.005
    loop_t = 0.125

    ik = IKHelper()

    if args.primitive == 'push' or args.primitive == 'pull':
        execute_thread = threading.Thread(target=_yumi_execute_arm,
                                          args=(yumi, active_arm))
    else:
        execute_thread = threading.Thread(target=_yumi_execute_both,
                                          args=(yumi,))
    execute_thread.daemon = True
    execute_thread.start()

    for plan_number, plan_dict in enumerate(initial_plan):

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

            for i in range(aligned_right.shape[0]):
                # r_pos = aligned_right[i, :]
                # l_pos = aligned_left[i, :]
                planned_r = aligned_right[i, :]
                planned_l = aligned_left[i, :]

                if is_in_contact(yumi, box_id):
                    # seed = {}
                    # seed['right'] = planned_r
                    # seed['left'] = planned_l

                    # joints = greedy_replan(
                    #     yumi, active_arm, box_id,
                    #     args.primitive, object_pose2_world,
                    #     args.config_package_path, ik, seed, i, plan_number)

                    # r_pos = joints['right']
                    # l_pos = joints['left']
                    r_pos = planned_r.tolist()
                    l_pos = planned_l.tolist()
                else:
                    r_pos = planned_r.tolist()
                    l_pos = planned_l.tolist()

                # embed()
                joint_lock.acquire()
                both_pos = r_pos + l_pos
                joint_lock.release()
                time.sleep(loop_t)
                # start = time.time()
                # while time.time() - start < loop_t:
                #     yumi.arm.set_jpos(np.hstack((r_pos, l_pos)), wait=False)
                #     time.sleep(sleep_t)
                # yumi.arm.set_jpos(np.hstack((r_pos, l_pos)), wait=True)
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
                # j_pos = point.positions
                planned_pos = point.positions
                if is_in_contact(yumi, box_id):
                    seed = {}
                    seed[active_arm] = planned_pos
                    seed[unactive_arm] = yumi.arm.arm_dict[unactive_arm].get_jpos()

                    joints = greedy_replan(
                        yumi, active_arm, box_id,
                        args.primitive, object_pose2_world,
                        args.config_package_path, ik, seed, i)

                    j_pos = joints[active_arm]
                else:
                    j_pos = planned_pos

                joint_lock.acquire()
                single_pos[active_arm] = j_pos
                joint_lock.release()
                time.sleep(loop_t)

                # start = time.time()
                # # yumi.arm.set_jpos(j_pos, arm=active_arm, wait=True)

                # while time.time() - start < loop_t:
                #     yumi.arm.set_jpos(j_pos, arm=active_arm, wait=False)
                #     time.sleep(sleep_t)
                
                # if i == len(traj.points)/2:
                #     perturb_box(yumi, box_id, [-0.025, -0.0125, 0.0])


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
