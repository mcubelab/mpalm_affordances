import numpy as np
import rospy
import roshelper
import helper
import ik.ik_helper
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import tf.transformations as tfm
from copy import deepcopy

def joints_from_poses(pose_left, pose_right, qf):
    if len(qf[0]) == 7:
        plan_dict_left = ik.ik_helper.joints_from_pose(pose_left, "l", seed=qf[0])
    else:
        plan_dict_left = ik.ik_helper.joints_from_pose(pose_left, "l", seed=None)
    if len(qf[1]) == 7:
        plan_dict_right = ik.ik_helper.joints_from_pose(pose_right, "r", seed=qf[1])
    else:
        plan_dict_right = ik.ik_helper.joints_from_pose(pose_right, "r", seed=None)
    return plan_dict_left, plan_dict_right

def raise_poses(poses, dz, arm=['r', 'l']):
    pose_left = deepcopy(poses[0])
    pose_right = deepcopy(poses[1])
    if 'l' in arm:
        pose_left.pose.position.z += dz
    if 'r' in arm:
        pose_right.pose.position.z += dz

    return [pose_left, pose_right]

def move_cart(desired_pose, plan_previous, primitive, speed=0.5, type='SetTraj', plan_name=None, N=100):
    #1. initialize variables
    is_execute = True
    out_dict = {}
    out_dict['is_failed_ik'] = False
    initial_pose = plan_previous['poses'][-1]
    object_pose = plan_previous['x_star'][-1]
    final_pose = desired_pose
    #2. interpolate poses individually (left and right)
    pose_left_interpolate_left = roshelper.interpolate_pose(initial_pose[0], final_pose[0], N)
    pose_left_interpolate_right = roshelper.interpolate_pose(initial_pose[1], final_pose[1], N)
    #3. Return robot plan
    out_dict['type'] = type
    out_dict['primitive'] = primitive
    out_dict['speed'] = speed
    out_dict['name'] = plan_name
    out_dict['is_replan'] = False
    out_dict['is_state_estimation'] = False
    out_dict['poses'] = [[pose_left_interpolate_left[x], pose_left_interpolate_right[x]] for x in
                         range(len(pose_left_interpolate_left))]
    # out_dict['poses_final'] = [final_pose[0], final_pose[1]]
    out_dict['x_star'] = [object_pose] * N
    out_dict['t_star'] = list(np.linspace(0, 1, num=N, endpoint=False))
    out_dict['is_execute'] = is_execute
    out_dict['replan_type'] = None
    return out_dict

def plan_sequence_poses(pose_list=[], primitive='', speed_list=[], name_list=[], previous_plan=None, check_collisions=True):
    robot_plan = []
    for counter in range(len(pose_list)):
        plan = move_cart(pose_list[counter],
                                         previous_plan,
                                         primitive,
                                         speed_list[counter],
                                         'SetCart',
                                         name_list[counter])
        plan['check_collisions'] = check_collisions
        robot_plan.append(plan)
        previous_plan = plan
    return robot_plan

def make_constant_plan(plan, plan_previous, N=11):
    plan['index_traj'] = []
    plan['object_traj'] = []
    for counter in range(N):
        try:
            plan['joint_traj'].points[counter]
        except:
            plan['joint_traj'].points.append(deepcopy(plan['joint_traj'].points[0]))
        plan['joint_traj'].points[counter].positions = deepcopy(plan['joint_traj'].points[0].positions)
        plan['joint_traj'].points[counter].time_from_start =  deepcopy(plan_previous['joint_traj'].points[counter].time_from_start)
        plan['index_traj'].append([counter * 100 / N, counter * 100 / N, 0.])
        plan['object_traj'].append(plan_previous['object_traj'][-1])
    return plan

def move_cart_synchro(final_pose, grasp_width, plan_previous, speed=1.0, primitive='', time=1, plan_name=None, N=100, is_replan=False, primitive_obj=None, is_state_estimation=False, is_state_estimation_initial=False,):
    #1. initialize variables
    joint_list = []
    out_dict = {}
    is_execute = True
    poses_inital = plan_previous['poses'][-1]
    object_pose = plan_previous['x_star'][-1]
    pose_initial_left_world = poses_inital[0]
    pose_final_left_world = final_pose
    # 2. define object pose relative to gripper frame
    pose_object_rel_gripper = roshelper.convert_reference_frame(object_pose,
                                                                pose_initial_left_world,
                                                                roshelper.unit_pose("yumi_body"),
                                                                frame_id="gripper_left")

    #2. interpolate gripper left pose trajectory
    pose_left_interpolate = roshelper.interpolate_pose(pose_initial_left_world, pose_final_left_world, N)
    #3. Loop through gripper poses and compute object poses
    object_pose_list = []
    pose_list = []
    for pose_left_world in pose_left_interpolate:
        pose_list_world = align_arm_poses(pose_left_world, grasp_width)
        pose_object_rel_world = roshelper.convert_reference_frame(pose_object_rel_gripper,
                                                        roshelper.unit_pose("yumi_body"),
                                                        pose_left_world,
                                                        frame_id="yumi_body")
        #4. append to list
        object_pose_list.append(pose_object_rel_world)
        pose_list.append([pose_list_world[0], pose_list_world[1]])
    #5. return final plan
    out_dict['poses'] = pose_list
    out_dict['poses_final'] = pose_list[-1]
    out_dict['type'] = 'SetTraj'
    out_dict['primitive'] = primitive
    out_dict['grasp_width'] = grasp_width
    out_dict['primitive_obj'] = primitive_obj
    out_dict['x_star'] = object_pose_list
    out_dict['speed'] = speed
    out_dict['time'] = time
    out_dict['is_replan'] = is_replan
    out_dict['replan_type'] = None
    out_dict['is_state_estimation'] = is_state_estimation
    out_dict['is_state_estimation_initial'] = is_state_estimation_initial
    out_dict['name'] = plan_name
    out_dict['t_star'] = list(np.linspace(0, time, num=N, endpoint=False))
    out_dict['is_execute'] = is_execute
    return out_dict

def move_object_synchro(rotation_base_initial, rotation_base_final, plan_previous, placement_list=None, delta_spring=0, anchor_offset=[], speed=0.5, primitive='', time=1, plan_name=None, N=100, planner=None, is_replan=False, is_state_estimation=False, is_state_estimation_initial=False, primitive_obj=None):
    #1. initialize variables
    out_dict = {}
    is_collision = False
    poses_inital = plan_previous['poses'][-1]
    object_pose_initial = plan_previous['x_star'][-1]
    pose_initial_left_offset_world = poses_inital[0]
    pose_initial_left_centered_world = offset_local_pose(pose_initial_left_offset_world, -np.array(anchor_offset))
    pose_initial_right_world = poses_inital[1]
    #interpolate object motion (rotate about rotate_base, i.e. corner of the object)
    pose_rotation_base_interpolate = roshelper.interpolate_pose(rotation_base_initial, rotation_base_final, N)

    #2. determinal relative poses between objects and rotate_base frame
    pose_object_rel_rotation_base, pose_gripper_left_rel_rotation_base, pose_gripper_right_rel_rotation_base = \
        roshelper.convert_list_reference_frames([object_pose_initial,
                                                 pose_initial_left_centered_world,
                                                 pose_initial_right_world],
                                                  [rotation_base_initial]*3,
                                                  [roshelper.unit_pose("yumi_body")]*3,
                                                  ["rotation_base"]*3)
    #4 loop through object poses and compute gripper poses
    gripper_pose_list = []
    object_pose_list = []
    angle_left_vec = np.linspace(0,90,len(pose_rotation_base_interpolate))
    angle_right_vec = np.linspace(3,3,len(pose_rotation_base_interpolate))

    for counter, pose_rotation_base_world in enumerate(pose_rotation_base_interpolate):
        #Find nominal gripper position in world based on object pose
        pose_gripper_left_world_tmp, \
        pose_gripper_right_world_tmp, \
        pose_object_world_tmp = roshelper.convert_list_reference_frames(
                                                        [pose_gripper_left_rel_rotation_base, pose_gripper_right_rel_rotation_base,pose_object_rel_rotation_base],
                                                        [roshelper.unit_pose("yumi_body")]*3,
                                                        [pose_rotation_base_world]*3,
                                                        ["yumi_body"]*3)
        #rotate gripper to desired orientation (about z gripper axis)
        pose_gripper_left_rotated_world, pose_gripper_right_rotated_world = rotate_gripper_poses([pose_gripper_left_world_tmp, pose_gripper_right_world_tmp],
                                                                                                 [[0,0,angle_left_vec[counter] * np.pi/180], [0,0,angle_right_vec[counter] * np.pi/180]])
        pose_gripper_left_rotated_world = offset_local_pose(pose_gripper_left_rotated_world, np.array(anchor_offset))
        # check for collision of left anchor:
        # if is_collision==False:
        pose_gripper_left_rotated_world, is_collision = avoid_collision(pose_gripper_left_rotated_world,
                                                          planner,
                                                          arm="l",
                                                          tol=0.003,
                                                          axis=[-1,0,0])
            # pose_gripper_left_safe = pose_gripper_left_rotated_world
        # else:
        #     pose_gripper_left_rotated_world = pose_gripper_left_safe

        object_pose_list.append(pose_object_world_tmp)
        gripper_pose_list.append([pose_gripper_left_rotated_world, pose_gripper_right_rotated_world])
    #5. return final plan
    out_dict['poses'] = gripper_pose_list
    out_dict['poses_final'] = gripper_pose_list[-1]
    out_dict['type'] = 'SetTraj'
    out_dict['primitive'] = primitive
    out_dict['primitive_obj'] = primitive_obj
    out_dict['x_star'] = object_pose_list
    out_dict['speed'] = speed
    out_dict['time'] = time
    out_dict['delta_spring'] = time
    out_dict['placement_list'] = placement_list
    out_dict['is_replan'] = is_replan
    out_dict['replan_type'] = None
    out_dict['is_state_estimation'] = is_state_estimation
    out_dict['is_state_estimation_initial'] = is_state_estimation_initial
    out_dict['name'] = plan_name
    out_dict['t_star'] = list(np.linspace(0, time, num=N, endpoint=False))
    return out_dict

def avoid_collision(pose_gripper, planner, arm="l", tol=0.003, axis=[-1,0,0]):
    is_collision = False
    pose_gripper_safe = deepcopy(pose_gripper)
    pose_gripper_safe.pose.position.z -= 0.006
    while planner.check_collisions_with_table(pose_gripper_safe, arm="l"):
        pose_gripper_safe = offset_local_pose(pose_gripper_safe, np.array(axis)*tol)
        is_collision = True
    pose_gripper = deepcopy(pose_gripper_safe)
    pose_gripper.pose.position.z += 0.006
    return pose_gripper, is_collision

def move_joints(desired_joints, plan_previous, primitive, speed=0.5, plan_name=None, N=100, check_collisions = True):
    desired_pose = [ik.ik_helper.compute_fk(desired_joints[0], "l"),
                    ik.ik_helper.compute_fk(desired_joints[1], "r")]
    out_dict = move_cart(desired_pose,
                         plan_previous,
                         primitive,
                         speed=speed,
                         type='SetJoints',
                         plan_name=plan_name,
                         N=N)
    out_dict['check_collisions'] = check_collisions
    out_dict['joints'] = [desired_joints]
    return out_dict

def offset_grasp(pose_stamped_world_frame, dist_grasp, offset, is_symmetric=True):
    '''Create new gripper poses offset from nominal gripper pose list'''

    pose_list_world = align_arm_poses(pose_stamped_world_frame[0], dist_grasp)
    offset_pose_list_world = offset_pose_list(pose_list_world, offset, is_symmetric)
    return offset_pose_list_world

def offset_local_pose(pose_world, offset):
    #1. convert to gripper reference frame
    pose_gripper = roshelper.convert_reference_frame(pose_world,
                                                         pose_world,
                                                         roshelper.unit_pose(frame_id="yumi_body"),
                                                         frame_id = "gripper_frame")

    #3. add offset to grasp poses in gripper frames
    pose_gripper.pose.position.x += offset[0]
    pose_gripper.pose.position.y += offset[1]
    pose_gripper.pose.position.z += offset[2]
    #4. convert back to world frame
    pose_new_world = roshelper.convert_reference_frame(pose_gripper,
                                                         roshelper.unit_pose(frame_id="yumi_body"),
                                                         pose_world,
                                                         frame_id = "yumi_body")
    return pose_new_world

def rotate_local_pose(pose_world, offset):
    angle_x = offset[0]
    angle_y = offset[1]
    angle_z = offset[2]
    pose_transform_tmp = roshelper.pose_from_matrix(tfm.euler_matrix(angle_x, angle_y, angle_z, 'sxyz'),
                                                frame_id="tmp")

    pose_rotated_world = roshelper.transform_body(pose_world, pose_transform_tmp)
    return pose_rotated_world

def rotate_gripper_poses(pose_world_list, offset_list):
    pose_rotated_world_list = []
    for i, pose_world in enumerate(pose_world_list):
            pose_rotated_world_list.append(rotate_local_pose(pose_world, offset_list[i]))
    return pose_rotated_world_list

def offset_gripper_poses(pose_stamped_world_frame, offset, is_offset=[True, True]):
    '''Create new gripper poses offset from nominal gripper pose list'''
    pose_new_world_list = []
    for i, pose_world in enumerate(pose_stamped_world_frame):
        if is_offset[i]:
            pose_new_world_list.append(offset_local_pose(pose_world, offset))
        else:
            pose_new_world_list.append(pose_world)
    return pose_new_world_list

def rotate_quat_y(pose):
    '''set orientation of right gripper as a mirror reflection of left gripper about y-axis'''
    quat = pose.pose.orientation
    R = tfm.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
    z_vec = R[0:3, 2]  # np.cross(x_vec, y_vec)
    y_vec = -R[0:3, 1]  # normal
    x_vec = np.cross(y_vec, z_vec)  # np.array([0,0,-1])
    x_vec = x_vec / np.linalg.norm(x_vec)
    y_vec = y_vec / np.linalg.norm(y_vec)
    z_vec = z_vec / np.linalg.norm(z_vec)
    # Normalized object frame
    hand_orient_norm = np.vstack((x_vec, y_vec, z_vec))
    hand_orient_norm = hand_orient_norm.transpose()
    quat = roshelper.mat2quat(hand_orient_norm)
    return quat

def align_arm_poses(pose, grasp_dist):
    '''Set position of right arm with respect to left arm'''
    T_left_world = roshelper.matrix_from_pose(pose)
    pose_right_left = PoseStamped()
    pose_right_left.header.frame_id = pose.header.frame_id
    pose_right_left.pose.position.y = - grasp_dist
    quat = rotate_quat_y(pose_right_left)
    pose_right_left.pose.orientation.x = quat[0]
    pose_right_left.pose.orientation.y = quat[1]
    pose_right_left.pose.orientation.z = quat[2]
    pose_right_left.pose.orientation.w = quat[3]
    T_right_left = roshelper.matrix_from_pose(pose_right_left)
    T_right_world = np.matmul(T_left_world, T_right_left)
    pose_right = roshelper.pose_from_matrix(T_right_world, type="PoseStamped")
    return [pose, pose_right]

def offset_pose_list(pose_list, offset = [0,0,0], is_symmetric=True):
    '''Offset pose_list by translation offset'''
    pose_offset_list = []
    arm_list = ['l', 'r']
    counter=0
    for pose, arm in zip(pose_list, arm_list):
        pose_stamped_hand = roshelper.unit_pose("tip_" + arm)
        if is_symmetric:
            pose_stamped_hand.pose.position.y =+ offset[1]
        else:
            if arm == 'r':
                pose_stamped_hand.pose.position.y =+ offset[1]
        if arm=='l':
            pose_stamped_hand.pose.position.x = + offset[0]
            pose_stamped_hand.pose.position.z = + offset[2]
        else:
            pose_stamped_hand.pose.position.x = - offset[0]
            pose_stamped_hand.pose.position.z = - offset[2]
        pose_stamped_world = roshelper.convert_reference_frame(pose_stamped_hand,
                                                                  roshelper.unit_pose("yumi_body"),
                                                                  pose)
        pose_offset_list.append(pose_stamped_world)
        counter+=1
    return pose_offset_list

def go_home_grasping(object_pose=None, speed=2., N=10, time=2):
    from sensor_msgs.msg import JointState
    #1. define final joint position
    final_joint_left = helper.yumi2robot_joints(rospy.get_param('grasping_left/yumi_convention/joints'), deg=True)
    final_joint_right = helper.yumi2robot_joints(rospy.get_param('grasping_right/yumi_convention/joints'), deg=True)

    #2. define initial (current) joint position
    if rospy.get_param('/have_robot'):
        if rospy.get_param('/egm_mode'):
            initial_joint_left_msg = rospy.wait_for_message('measured_joints_left', JointState)
            initial_joint_right_msg = rospy.wait_for_message('measured_joints_right', JointState)
            initial_joint_left = helper.yumi2robot_joints(initial_joint_left_msg.position)
            initial_joint_right = helper.yumi2robot_joints(initial_joint_right_msg.position)
        else:
            initial_joint_left = final_joint_left
            initial_joint_right = final_joint_right
    else:
        joint_states_msg = rospy.wait_for_message('/joint_states', JointState)
        initial_joint_left = joint_states_msg.position[7:14]
        initial_joint_right = joint_states_msg.position[0:7]

    #3. perform linear interpolation
    joint_left_interpolate = helper.linspace_array(initial_joint_left, final_joint_left, N)
    joint_right_interpolate = helper.linspace_array(initial_joint_right, final_joint_right, N)

    #4. loop through all joints and compute poses
    joints_list = []
    poses_list = []
    for joint_left, joint_right in zip(joint_left_interpolate, joint_right_interpolate):
        #5. compute fk
        pose_left = ik.ik_helper.compute_fk(joint_left, arm="l")
        pose_right = ik.ik_helper.compute_fk(joint_right, arm="r")
        #6. append joints and poses to list
        joints_list.append([joint_left, joint_right])
        poses_list.append([pose_left, pose_right])
    plan_dict = {}
    plan_dict['poses'] = poses_list
    plan_dict['poses_final'] = poses_list[-1]
    plan_dict['type'] = 'SetTraj'
    plan_dict['primitive'] = 'go_home'
    plan_dict['speed'] = speed
    plan_dict['time'] = time
    plan_dict['t_star'] = list(np.linspace(0, 1, num=N, endpoint=False))
    plan_dict['is_execute'] = True
    if object_pose is not None:
        plan_dict['x_star'] = [object_pose] * N
    plan_dict['joints'] = joints_list
    plan_dict['joints_final'] = joints_list[-1]
    return plan_dict

