# import copy
# import sys, os
# from helper import util, planning_helper, collisions
# import numpy as np
# from example_config_cfg import get_cfg_defaults
# import argparse
import os
import sys
import numpy as np
import copy

from rpo_planning.utils import common as util
from rpo_planning.utils.planning import primitive as planning_helper
from rpo_planning.utils.planning import collision as collisions

def grasp_planning(object, object_pose1_world, object_pose2_world,
                   palm_pose_l_object, palm_pose_r_object, N=200, init=True):
    """
    Main grasping primitive function. Return a plan that contains the pose
    trajectories of the object and palms to achieve desired object
    reconfiguration.

    Args:
        object (collisions.CollisionBody): that contains the geometry of
            the object (not currently used)
        object_pose1_world (util.PoseStamped): Initial object pose in world
            frame.
        object_pose2_world (util.PoseStamped): Final object pose in world
            frame.
        palm_pose_l_object (util.PoseStamped): Left palm pose in object
            frame.
        palm_pose_r_object (util.PoseStamped): Right palm pose in object
            frame.

    Returns:
        list: (list of dict with keys)
            palm_poses_r_world (list of util.PoseStamped): Trajectory of right
                palm poses in world frame
            palm_poses_l_world (list of util.PoseStamped): Trajectory of left
                palm poses in world frame
            object_poses_world (util.PoseStamped): Trajectory of object poses
                in world frame
            primitive (util.PoseStamped): Name of primitive (i.e., 'grasping')
            name (util.PoseStamped): Name of plan
            t (util.PoseStamped): list of timestamps associated with each pose
            N (util.PoseStamped): Number of keypoints in the plan
                (i.e., len(plan_dict['t'])
    """
    primitive_name = 'grasping'
    # 0. get initial palm poses in world frame
    palm_poses_initial_world = planning_helper.palm_poses_from_object(
            object_pose=object_pose1_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object)

    grasp_width = planning_helper.grasp_width_from_palm_poses(
        palm_pose_l_object, palm_pose_r_object)
    # grasp_width = grasp_width/1.025
    # grasp_width = 0.086 - 0.006
    # grasp_width = 0.086
    # grasp_width = 0.05206
    # grasp_width = 0.135
    # print("grasp width: " + str(grasp_width))

    # 1. get lifted object poses
    object_pose_lifted_world = copy.deepcopy(object_pose1_world)

    if init:
        object_pose_lifted_world.pose.position.z += 0.05
    # object_pose_lifted_world.pose.position.z += 0.05
    object_pose2_world.pose.position.z += 0.0025

    # 2. get lifted palm poses
    palm_poses_lifted_world = planning_helper.palm_poses_from_object(
        object_pose=object_pose_lifted_world,
        palm_pose_l_object=palm_pose_l_object,
        palm_pose_r_object=palm_pose_r_object)

    # 3. get rotated object pose
    object_pose_rotated_world = copy.deepcopy(object_pose2_world)

    # if init:
    #     object_pose_rotated_world.pose.position.z += 0.05
    object_pose_rotated_world.pose.position.z += 0.05

    palm_poses_rotated_world = planning_helper.palm_poses_from_object(
        object_pose=object_pose_rotated_world,
        palm_pose_l_object=palm_pose_l_object,
        palm_pose_r_object=palm_pose_r_object)

    # 4. get final configuration
    palm_poses_final_world = planning_helper.palm_poses_from_object(
        object_pose=object_pose2_world,
        palm_pose_l_object=palm_pose_l_object,
        palm_pose_r_object=palm_pose_r_object)

    # 3. generate pose plans
    # 3.1. initialize plan
    initial_plan = planning_helper.initialize_plan(
        palm_poses_initial=palm_poses_initial_world,
        object_pose_initial=object_pose1_world,
        primitive=primitive_name,
        plan_name='initial_config')

    # 3.2. lift the object
    lift_plan = planning_helper.move_cart_synchro(
        palm_poses_final=palm_poses_lifted_world,
        grasp_width=grasp_width,
        plan_previous=initial_plan,
        primitive=primitive_name,
        plan_name='lift_object',
        N=10)

    # 3.3. rotate the object
    rotate_plan = planning_helper.move_cart_synchro(
        palm_poses_final=palm_poses_rotated_world,
        grasp_width=grasp_width,
        plan_previous=lift_plan,
        primitive=primitive_name,
        plan_name='rotate_object_final',
        N=N/2,
        is_replan=True)

    # 3.4. place the object
    place_plan = planning_helper.move_cart_synchro(
        palm_poses_final=palm_poses_final_world,
        grasp_width=grasp_width,
        plan_previous=rotate_plan,
        primitive=primitive_name,
        plan_name='place_object',
        N=20)
    return [lift_plan] + [rotate_plan] + [place_plan]


def grasp_planning_wf(palm_pose_l_world, palm_pose_r_world,
                      transformation, N=200, init=True):
    """
    Main grasping primitive function. Return a plan that contains the pose
    trajectories of the object and palms to achieve desired object
    reconfiguration.

    Args:
        object (collisions.CollisionBody): that contains the geometry of
            the object (not currently used)
        object_pose1_world (util.PoseStamped): Initial object pose in world
            frame.
        object_pose2_world (util.PoseStamped): Final object pose in world
            frame.
        palm_pose_l_object (util.PoseStamped): Left palm pose in object
            frame.
        palm_pose_r_object (util.PoseStamped): Right palm pose in object
            frame.

    Returns:
        list: (list of dict with keys)
            palm_poses_r_world (list of util.PoseStamped): Trajectory of right
                palm poses in world frame
            palm_poses_l_world (list of util.PoseStamped): Trajectory of left
                palm poses in world frame
            object_poses_world (util.PoseStamped): Trajectory of object poses
                in world frame
            primitive (util.PoseStamped): Name of primitive (i.e., 'grasping')
            name (util.PoseStamped): Name of plan
            t (util.PoseStamped): list of timestamps associated with each pose
            N (util.PoseStamped): Number of keypoints in the plan
                (i.e., len(plan_dict['t'])
    """
    primitive_name = 'grasping'
    # rotate_height = 0.05
    rotate_height = 0.075
    # rotate_height = 0.1
    # 0. get initial palm poses in world frame
    palm_poses_initial_world = [palm_pose_l_world, palm_pose_r_world]

    grasp_width = planning_helper.grasp_width_from_palm_poses(
        palm_pose_l_world, palm_pose_r_world)

    # 1. get lifted palm poses in world frame
    palm_poses_lifted_world = []
    for pose in palm_poses_initial_world:
        lifted_pose = copy.deepcopy(pose)
        lifted_pose.pose.position.z += rotate_height
        palm_poses_lifted_world.append(lifted_pose)

    # 2. get final configuration
    palm_poses_final_world = []
    for pose in palm_poses_initial_world:
        final_pose = util.transform_pose(
            pose_source=pose,
            pose_transform=transformation)
        final_pose.pose.position.z += 0.0025
        palm_poses_final_world.append(final_pose)

    # 3. get rotated + lifted palm pose in world frame
    palm_poses_rotated_world = []
    for pose in palm_poses_final_world:
        rotated_pose = copy.deepcopy(pose)
        rotated_pose.pose.position.z += rotate_height
        palm_poses_rotated_world.append(rotated_pose)

    # 3. generate pose plans
    # 3.1. initialize plan
    initial_plan = planning_helper.initialize_plan_wf(
        palm_poses_initial=palm_poses_initial_world,
        transformation=transformation,
        primitive=primitive_name,
        plan_name='initial_config')

    # 3.2. lift the object
    lift_plan = planning_helper.move_cart_synchro_wf(
        palm_poses_final=palm_poses_lifted_world,
        grasp_width=grasp_width,
        plan_previous=initial_plan,
        primitive=primitive_name,
        plan_name='lift_object',
        N=10)

    # 3.3. rotate the object
    rotate_plan = planning_helper.move_cart_synchro_wf(
        palm_poses_final=palm_poses_rotated_world,
        grasp_width=grasp_width,
        plan_previous=lift_plan,
        primitive=primitive_name,
        plan_name='rotate_object_final',
        N=N/2,
        is_replan=True)

    # 3.4. place the object
    place_plan = planning_helper.move_cart_synchro_wf(
        palm_poses_final=palm_poses_final_world,
        grasp_width=grasp_width,
        plan_previous=rotate_plan,
        primitive=primitive_name,
        plan_name='place_object',
        N=20)
    return [lift_plan] + [rotate_plan] + [place_plan]


def levering_planning(object, object_pose1_world, object_pose2_world,
                      palm_pose_l_object, palm_pose_r_object,
                      rotation_center_pose_world=None,
                      anchor_offset=[-0.01, 0,0], gripper_name=None,
                      table_name=None, N=100, planner=None):
    """
    Main levering primitive function. Return a plan that contains
    the pose trajectories of the object and palms to achieve
    desired object reconfiguration.

    Args:
        object (collisions.CollisionBody): that contains the geometry
            of the object. If object=None used, collisions will occur
            between palms and table.
        object_pose1_world (util.PoseStamped): Initial object pose
            in world frame.
        object_pose2_world (util.PoseStamped): Final object pose
            in world frame.
        palm_pose_l_object (util.PoseStamped): Left palm pose in
            object frame.
        palm_pose_r_object (util.PoseStamped): Right palm pose in
            object frame.

    Returns:
        list: (list of dict with keys)
            palm_poses_r_world (list of util.PoseStamped): Trajectory
                of right palm poses in world frame
            palm_poses_l_world (list of util.PoseStamped): Trajectory
                of left palm poses in world frame
            object_poses_world (util.PoseStamped): Trajectory of object
                poses in world frame
            primitive (util.PoseStamped): Name of primitive (i.e., 'grasping')
            name (util.PoseStamped): Name of plan
            t (util.PoseStamped): list of timestamps associated with
                each pose
            N (util.PoseStamped): Number of keypoints in the plan
                (i.e., len(plan_dict['t'])
    """
    primitive_name = 'levering'
    if gripper_name is not None and table_name is not None:
        collision_check = collisions.CheckCollisions(
            gripper_name=gripper_name,
            table_name=table_name)

    if rotation_center_pose_world is None:
        rotation_center_pose_world = planning_helper.rotation_center_from_object_poses(
            corners_object=object.trimesh.vertices,
            object_pose_initial=object_pose1_world,
            object_pose_final=object_pose2_world,)

    #0. get initial palm poses in world frame
    palm_poses_initial_world = planning_helper.palm_poses_from_object(
        object_pose=object_pose1_world,
        palm_pose_l_object=palm_pose_l_object,
        palm_pose_r_object=palm_pose_r_object)

    #1. get poses relative to rotation center
    object_pose_center = util.convert_reference_frame(
        pose_source=object_pose1_world,
        pose_frame_target=rotation_center_pose_world,
        pose_frame_source=util.unit_pose(),
        frame_id="rotation_center")

    palm_pose_l_offset_world = util.offset_local_pose(
        palm_poses_initial_world[0],
        -np.array(anchor_offset))

    palm_pose_l_center = util.convert_reference_frame(
        pose_source=palm_pose_l_offset_world,
        pose_frame_target=rotation_center_pose_world,
        pose_frame_source=util.unit_pose(),
        frame_id="rotation_center")

    palm_pose_r_center = util.convert_reference_frame(
        pose_source=palm_poses_initial_world[1],
        pose_frame_target=rotation_center_pose_world,
        pose_frame_source=util.unit_pose(),
        frame_id="rotation_center")

    #2. interpolation rotation center from initial to final pose
    object_pose_transform = util.get_transform(
        pose_frame_target=object_pose2_world,
        pose_frame_source=object_pose1_world)

    euler_angles = util.euler_from_pose(
        object_pose_transform)

    rotation_center_pose_final_world = util.transform_pose(
        pose_source=rotation_center_pose_world,
        pose_transform=object_pose_transform)

    rotation_center_pose_world_list = util.interpolate_pose(
        rotation_center_pose_world,
        rotation_center_pose_final_world,
        N=N)

    #. in body frame
    angle_left_vec = np.linspace(0, -euler_angles[0] * 180 / np.pi, len(rotation_center_pose_world_list))
    angle_right_vec = np.linspace(0, 0, len(rotation_center_pose_world_list))
    palm_poses_world_list = []
    palm_pose_l_world_list = []
    palm_pose_r_world_list = []
    palm_pose_l_object_list = []
    palm_pose_r_object_list = []
    object_pose_world_list = []

    for counter, rotation_center_pose_world in enumerate(rotation_center_pose_world_list):
        object_world_tmp = util.convert_reference_frame(
            pose_source=object_pose_center,
            pose_frame_target=util.unit_pose(),
            pose_frame_source=rotation_center_pose_world,
            frame_id="world")
        palm_pose_l_world_tmp = util.convert_reference_frame(
            pose_source=palm_pose_l_center,
            pose_frame_target=util.unit_pose(),
            pose_frame_source=rotation_center_pose_world,
            frame_id="world")
        palm_pose_r_world_tmp = util.convert_reference_frame(
            pose_source=palm_pose_r_center,
            pose_frame_target=util.unit_pose(),
            pose_frame_source=rotation_center_pose_world,
            frame_id="world")

        #4. rotate palms linearly as the object rotates
        palm_pose_l_world, palm_pose_r_world = util.rotate_local_pose_list(
            pose_world_list=[palm_pose_l_world_tmp, palm_pose_r_world_tmp],
            offset_list=[[0, 0, angle_left_vec[counter] * np.pi / 180],
            [0, 0, angle_right_vec[counter] * np.pi / 180]])
        palm_pose_l_world = util.offset_local_pose(palm_pose_l_world,
                                                    np.array(anchor_offset))

        #5. Continuously check for collisions between left palm and table (if collision, move palm up)
        if gripper_name is not None and table_name is not None:
            palm_pose_l_world = collision_check.avoid_collision(palm_pose_l_world,
                                                                arm="l",
                                                                tol=0.001,
                                                                axis=[-1, 0, 0])

        #6. convert palm poses to object frame
        palm_pose_l_object = util.convert_reference_frame(pose_source=palm_pose_l_world,
                                                          pose_frame_target=object_world_tmp,
                                                          pose_frame_source=util.unit_pose(),
                                                          frame_id="object")
        palm_pose_r_object = util.convert_reference_frame(pose_source=palm_pose_r_world,
                                                          pose_frame_target=object_world_tmp,
                                                          pose_frame_source=util.unit_pose(),
                                                          frame_id="object")

        #7. save variables
        palm_poses_world_list.append([palm_pose_l_world, palm_pose_r_world])
        palm_pose_l_world_list.append(palm_pose_l_world)
        palm_pose_r_world_list.append(palm_pose_r_world)
        palm_pose_l_object_list.append(palm_pose_l_object)
        palm_pose_r_object_list.append(palm_pose_r_object)
        object_pose_world_list.append(object_world_tmp)

    #6. return final plan
    plan_dict = {}
    plan_dict['palm_poses_world'] = palm_poses_world_list
    plan_dict['palm_pose_l_world'] = palm_pose_l_world_list
    plan_dict['palm_pose_r_world'] = palm_pose_r_world_list
    plan_dict['palm_pose_l_object'] = palm_pose_l_object_list
    plan_dict['palm_pose_r_object'] = palm_pose_r_object_list
    plan_dict['primitive'] = primitive_name
    plan_dict['object_poses_world'] = object_pose_world_list
    plan_dict['name'] = 'rotate_object'
    plan_dict['t'] = list(np.linspace(0, 1, num=N, endpoint=False))
    plan_dict['N'] = N
    return [plan_dict]


def pushing_planning(object, object_pose1_world, object_pose2_world,
                     palm_pose_l_object, palm_pose_r_object, arm='r', pusher_angle=0.0):
    """
    Main pushing primitive function. Return a plan that contains the
    pose trajectories of the object and palms to achieve desired object reconfiguration.

    Args:
    object (collisions.CollisionBody): that contains the geometry of the object.
        (Not currently used)
    object_pose1_world (util.PoseStamped): Initial object pose in world frame.
    object_pose2_world (util.PoseStamped): Final object pose in world frame.
    palm_pose_l_object (util.PoseStamped): Left palm pose in object frame.
    palm_pose_r_object (util.PoseStamped): Right palm pose in object frame.

    Returns:
        list (list of dict with: keys)
            palm_poses_r_world (list of util.PoseStamped): Trajectory of right
                palm poses in world frame
            palm_poses_l_world (list of util.PoseStamped): Trajectory of left
                palm poses in world frame
            object_poses_world (util.PoseStamped): Trajectory of object poses
                in world frame
            primitive (util.PoseStamped): Name of primitive (i.e., 'grasping')
            name: (util.PoseStamped): Name of plan
            t (util.PoseStamped): list of timestamps associated with each pose
            N (util.PoseStamped): Number of keypoints in the plan
                (i.e., len(plan_dict['t'])
    """
    primitive_name = 'pushing'
    #0. get initial palm poses in world frame
    palm_poses_initial_world = planning_helper.palm_poses_from_object(
        object_pose=object_pose1_world,
        palm_pose_l_object=palm_pose_l_object,
        palm_pose_r_object=palm_pose_r_object)

    #1. Convert pose to 2d pose
    object_initial_planar_pose = planning_helper.get_2d_pose(object_pose1_world)
    object_final_planar_pose = planning_helper.get_2d_pose(object_pose2_world)

    #2.
    configurations_transformed, N_star, object_pose_2d_list, t_star = planning_helper.dubins_trajectory(
        q0=object_initial_planar_pose,
        qf=object_final_planar_pose,
        radius=0.125,
        velocity_real=0.05,
        step_size=0.015,
        contact_angle=pusher_angle)
    # 3. iterate through trajectory and compute robot i)poses and ii)joints
    object_pose_world_list = []
    palm_poses_world_list = []
    palm_pose_l_world_list = []
    palm_pose_r_world_list = []
    for counter, object_pose_2d in enumerate(object_pose_2d_list):
    # 4. get 3d object pose from 2d
        object_pose__world = planning_helper.get3dpose_object(
            pose2d=object_pose_2d,
            pose3d_nominal=object_pose1_world)
        if arm=='r':
            palm_pose_l_world = palm_poses_initial_world[0]
            palm_pose_r_world = util.convert_reference_frame(
                pose_source=palm_pose_r_object,
                pose_frame_target=util.unit_pose(),
                pose_frame_source=object_pose__world,
                frame_id="yumi_body",
                )
        else:
            palm_pose_l_world = util.convert_reference_frame(
                pose_source=palm_pose_l_object,
                pose_frame_target=util.unit_pose(),
                pose_frame_source=object_pose__world,
                frame_id="yumi_body",
                )
            palm_pose_r_world = palm_poses_initial_world[1]
        object_pose_world_list.append(object_pose__world)
        palm_poses_world_list.append([palm_pose_l_world, palm_pose_r_world])
        palm_pose_l_world_list.append(palm_pose_l_world)
        palm_pose_r_world_list.append(palm_pose_r_world)
    #5. return final plan
    plan_dict = {}
    plan_dict['palm_poses_world'] = palm_poses_world_list
    plan_dict['palm_pose_l_world'] = palm_pose_l_world_list
    plan_dict['palm_pose_r_world'] = palm_pose_r_world_list
    plan_dict['primitive'] = primitive_name
    plan_dict['object_poses_world'] = object_pose_world_list
    plan_dict['name'] = 'push_object'
    plan_dict['t'] = t_star
    plan_dict['N'] = len(N_star)
    return [plan_dict]

def pushing_planning_wf(palm_pose_l_world, palm_pose_r_world, 
                        transformation, arm='r', G_xy=[0, 0], pusher_angle=np.pi/2, velocity_real=0.01):
    primitive_name = 'pushing'
    #0. get initial palm poses in world frame
    palm_poses_initial_world = [palm_pose_l_world, palm_pose_r_world]

    #1. Convert pose to 2d pose
    # initialize the object pose at the palm pose
    object_initial_pose = palm_pose_r_world if arm == 'r' else palm_pose_l_world

    # # account for approximate xy offset from object center of mass
    # object_initial_pose.pose.position.x -= G_xy[0]
    # object_initial_pose.pose.position.y -= G_xy[1] 

    # convert to 2d
    object_initial_planar_pose = planning_helper.get_2d_pose(object_initial_pose)
    object_final_planar_pose = planning_helper.get_2d_pose(util.transform_pose(object_initial_pose, transformation))

    #2.
    configurations_transformed, N_star, object_pose_2d_list, t_star = planning_helper.dubins_trajectory(
        q0=object_initial_planar_pose,
        qf=object_final_planar_pose,
        radius=0.125,
        velocity_real=velocity_real,
        step_size=0.015,
        contact_angle=pusher_angle)
    # 3. iterate through trajectory and compute robot i)poses and ii)joints
    object_pose_world_list = []
    palm_poses_world_list = []
    palm_pose_l_world_list = []
    palm_pose_r_world_list = []
    obj_pose_world_prev = object_initial_pose
    palm_pose_r_prev = palm_pose_r_world
    palm_pose_l_prev = palm_pose_l_world
    for counter, object_pose_2d in enumerate(object_pose_2d_list):
    # 4. get 3d object pose from 2d
        obj_pose_world_current = planning_helper.get3dpose_object(
            pose2d=object_pose_2d,
            pose3d_nominal=object_initial_pose)

        # get transform between consecutive poses, and apply consecutively to palm poses
        obj_transform = util.get_transform(obj_pose_world_current, obj_pose_world_prev)
        
        palm_pose_r_world = util.transform_pose(palm_pose_r_prev, obj_transform)
        palm_pose_l_world = util.transform_pose(palm_pose_l_prev, obj_transform)

        # if arm=='r':
        #     palm_pose_l_world = palm_poses_initial_world[0]
        #     palm_pose_r_world = util.convert_reference_frame(
        #         pose_source=palm_pose_r_object,
        #         pose_frame_target=util.unit_pose(),
        #         pose_frame_source=object_pose__world,
        #         frame_id="yumi_body",
        #         )
        # else:
        #     palm_pose_l_world = util.convert_reference_frame(
        #         pose_source=palm_pose_l_object,
        #         pose_frame_target=util.unit_pose(),
        #         pose_frame_source=object_pose__world,
        #         frame_id="yumi_body",
        #         )
        #     palm_pose_r_world = palm_poses_initial_world[1]
        palm_poses_world_list.append([palm_pose_l_world, palm_pose_r_world])
        palm_pose_l_world_list.append(palm_pose_l_world)
        palm_pose_r_world_list.append(palm_pose_r_world)

        palm_pose_r_prev = palm_pose_r_world
        palm_pose_l_prev = palm_pose_l_world
        obj_pose_world_prev = obj_pose_world_current
    #5. return final plan
    plan_dict = {}
    plan_dict['palm_poses_world'] = palm_poses_world_list
    plan_dict['palm_pose_l_world'] = palm_pose_l_world_list
    plan_dict['palm_pose_r_world'] = palm_pose_r_world_list
    plan_dict['primitive'] = primitive_name
    plan_dict['name'] = 'push'
    plan_dict['t'] = t_star
    plan_dict['N'] = len(N_star)
    return [plan_dict]

def pulling_planning(object, object_pose1_world, object_pose2_world,
                     palm_pose_l_object, palm_pose_r_object, arm='r', N=60):
    primitive_name = 'pulling'
    if arm=='r':
        palm_rel_object = palm_pose_r_object
        pose_l_nominal_world = util.convert_reference_frame(
            palm_pose_l_object,
            util.unit_pose(),
            object_pose1_world,
            "yumi_body")
    else:
        palm_rel_object = palm_pose_l_object
        pose_r_nominal_world = util.convert_reference_frame(
            palm_pose_r_object,
            util.unit_pose(),
            object_pose1_world,
            "yumi_body")
    #1. convert to planar poses
    palm_poses_list = []
    for counter, pose in enumerate([object_pose1_world, object_pose2_world]):
        # 4. compute gripper pose from object pose
        robot_pose = util.convert_reference_frame(palm_rel_object,
                                                   util.unit_pose(),
                                                   pose,
                                                   "yumi_body")
        # 5. append values
        if arm == "r":
            palm_poses_list.append([pose_l_nominal_world, robot_pose])
        else:
            palm_poses_list.append([robot_pose, pose_r_nominal_world])

    #interpolate poses
    object_pose_world_list = util.interpolate_pose(
        pose_initial=object_pose1_world,
        pose_final=object_pose2_world,
        N=N)
    palm_pose_l_world_list = util.interpolate_pose(
        pose_initial=palm_poses_list[0][0],
        pose_final=palm_poses_list[-1][0],
        N=N)
    palm_pose_r_world_list = util.interpolate_pose(
        pose_initial=palm_poses_list[0][1],
        pose_final=palm_poses_list[-1][1],
        N=N)
    poses_array = np.vstack((np.array(palm_pose_l_world_list),
                             np.array(palm_pose_r_world_list))).transpose()
    palm_poses_list = list(poses_array)
    #5. return final plan
    plan_dict = {}
    plan_dict['palm_poses_world'] = palm_poses_list
    plan_dict['palm_pose_l_world'] = palm_pose_l_world_list
    plan_dict['palm_pose_r_world'] = palm_pose_r_world_list
    plan_dict['primitive'] = primitive_name
    plan_dict['object_poses_world'] = object_pose_world_list
    plan_dict['name'] = 'pull'
    plan_dict['t'] = list(np.linspace(0, 1, num=N, endpoint=False))
    plan_dict['N'] = N
    return [plan_dict]


def pulling_planning_wf(palm_pose_l_world, palm_pose_r_world, 
                        transformation, arm='r', N=60):
    primitive_name = 'pulling'
    # 1. transform start pose to final pose
    palm_poses_list = []
    palm_poses_start = [palm_pose_l_world, palm_pose_r_world]
    palm_poses_final = []
    for pose in palm_poses_start:
        final_pose = util.transform_pose(
            pose,
            transformation
        )
        palm_poses_final.append(final_pose)
    palm_poses_list.append(palm_poses_start)
    palm_poses_list.append(palm_poses_final)

    # 2. interpolate
    palm_pose_l_world_list = util.interpolate_pose(
        pose_initial=palm_poses_list[0][0],
        pose_final=palm_poses_list[-1][0],
        N=N
    )
    palm_pose_r_world_list = util.interpolate_pose(
        pose_initial=palm_poses_list[0][1],
        pose_final=palm_poses_list[-1][1],
        N=N
    )
    poses_array = np.vstack((np.array(palm_pose_l_world_list),
                             np.array(palm_pose_r_world_list))).transpose()

    palm_poses_list = list(poses_array)

    # 3. return dict
    plan_dict = {}
    plan_dict['palm_poses_world'] = palm_poses_list
    plan_dict['palm_pose_l_world'] = palm_pose_l_world_list
    plan_dict['palm_pose_r_world'] = palm_pose_r_world_list
    plan_dict['N'] = N
    plan_dict['t'] = list(np.linspace(0, 1, num=N, endpoint=False))
    plan_dict['primitive'] = primitive_name
    return [plan_dict]


def main(args):
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
        gripper_name = args.config_package_path + 'descriptions/meshes/mpalm/mpalms_all_coarse.stl'
        table_name = args.config_package_path + 'descriptions/meshes/table/table_top.stl'

        manipulated_object = collisions.CollisionBody(
            args.config_package_path + 'descriptions/meshes/objects/realsense_box_experiments.stl')

        plan = levering_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            gripper_name=gripper_name,
            table_name=table_name
        )

    # 4. pulling primitive
    elif args.primitive == 'pull':
        plan = pulling_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            arm='r')

    if args.simulate:
        import simulation
        import rospy

        rospy.init_node("test")
        for i in range(10):
            simulation.visualize_object(
                object_pose1_world,
                filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
                name="/object_initial",
                color=(1., 0., 0., 1.),
                frame_id="/yumi_body",
                scale=(1., 1., 1.))
            simulation.visualize_object(
                object_pose2_world,
                filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
                name="/object_final",
                color=(0., 0., 1., 1.),
                frame_id="/yumi_body",
                scale=(1., 1., 1.))
            rospy.sleep(.1)
        simulation.simulate(plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_package_path',
                        type=str,
                         default='/home/anthony/repos/research/mpalm_affordances/catkin_ws/src/config/')
    parser.add_argument('--example_config_path', type=str, default='config')
    parser.add_argument('--primitive', type=str, default='push', help='which primitive to plan')
    parser.add_argument('--simulate', type=bool, default=True)
    args = parser.parse_args()
    main(args)
