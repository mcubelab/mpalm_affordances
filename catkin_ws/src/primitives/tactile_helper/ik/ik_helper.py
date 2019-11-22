#!/usr/bin/env python


from trac_ik_python.trac_ik import IK
import numpy as np
from tf.transformations import quaternion_from_euler
import tf, time
import tf.transformations as tfm
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import helper
from helper import util
import rospy
import moveit_msgs
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.srv import GetPositionIKRequest
from moveit_msgs.srv import GetPositionIKResponse

#define forward kinematics configuration
robot = URDF.from_parameter_server()
fk_solver_r = KDLKinematics(robot, "yumi_body", "yumi_tip_r")
fk_solver_l = KDLKinematics(robot, "yumi_body", "yumi_tip_l")

#define inverse kinematics configuration
ik_solver_r = IK("yumi_body", "yumi_tip_r")
ik_solver_l = IK("yumi_body", "yumi_tip_l")

ik_srv = rospy.ServiceProxy('/compute_ik', GetPositionIK)

def joints_from_pose_list(pose_list, seed_both_arms=None):
    arm_list = ['l', 'r']
    dict_ik_pose = {}
    dict_ik_pose['joints'] = []
    dict_ik_pose['is_execute'] = []
    for pose, arm in zip(pose_list, arm_list):
        # roshelper.handle_block_pose(pose.pose, self.br, "/yumi_body", "/gripper_" + arm)
        counter = 0
        if seed_both_arms==None:
            seed_both_arms = [1.2845762921016486, -1.728097616287859, -1.484556117795128, -0.060662408807661716,
                  0.9363691436943, 1.045354483503344, 2.9032784162076473,
                  -1.427503049876694, -1.7222228380256484, 1.1440022381582904, -0.034734670104648924,
                  -0.4423711522101721, 0.5704434127183402, -0.17426065383625167]
        while True:
            if arm == "l":
                seed = helper.helper.yumi2robot_joints(rospy.get_param('grasping_left/yumi_convention/joints'), deg=True)
            else:
                seed = helper.helper.yumi2robot_joints(rospy.get_param('grasping_right/yumi_convention/joints'), deg=True)

            # joints = compute_ik(helper.roshelper.pose_stamped2list(pose), seed, arm=arm)
            joints = compute_ik(
                util.pose_stamped2list(pose), seed, arm=arm)
            counter += 1
            if len(joints) > 2 or counter > 20:
                dict_ik_pose['joints'].append(joints)
                break
        if len(joints)==7:
            dict_ik_pose['is_execute'].append(True)
        else:
            dict_ik_pose['is_execute'].append(False)
    return dict_ik_pose


def joints_from_pose(pose, arm, seed=None):
    dict_ik_pose = {}
    counter = 0
    if seed is None:
        # seed_both_arms = [1.6186728267172805, -0.627957447708273, -2.181258817200459, 0.6194790456943977, 3.286937582146461, 0.7514464459671322, -3.038332444527036, -0.8759314115583274, -1.008775144531517, 1.5611336697750984, 0.5415813719525806, 2.4631588845619383, 0.5179680763672723, 1.5100636942652095]

        # seed_both_arms = [1.2845762921016486, -1.728097616287859, -1.484556117795128, -0.060662408807661716,
        #                   0.9363691436943, 1.045354483503344, 2.9032784162076473,
        #                   -1.427503049876694, -1.7222228380256484, 1.1440022381582904, -0.034734670104648924,
        #                   -0.4423711522101721, 0.5704434127183402, -0.17426065383625167]
        if arm == "l":
            seed = helper.helper.yumi2robot_joints(rospy.get_param('grasping_left/yumi_convention/joints'), deg=True)
            # seed = seed_both_arms[7:14]
        else:
            seed = helper.helper.yumi2robot_joints(rospy.get_param('grasping_right/yumi_convention/joints'), deg=True)
            # seed = seed_both_arms[0:7]
    # while True:
    # joints = compute_ik(helper.roshelper.pose_stamped2list(pose), seed, arm=arm)
    joints = compute_ik(
        util.pose_stamped2list(pose), seed, arm=arm)
        # counter += 1
    if len(joints) > 2:# or counter > 10:
        dict_ik_pose['joints'] = joints
            # break
    if len(joints) == 7:
        dict_ik_pose['is_execute'] = True
    else:
        dict_ik_pose['is_execute'] = False
    return dict_ik_pose

def compute_fk(joints, arm="r"):
    if arm=="r":
        matrix = fk_solver_r.forward(joints)
    else:
        matrix = fk_solver_l.forward(joints)
    translation = tfm.translation_from_matrix(matrix)
    quat = tfm.quaternion_from_matrix(matrix)
    pose_array = np.hstack((translation, quat))
    # return helper.roshelper.convert_pose_type(pose_array,
    #                   type_out="PoseStamped",
    #                   frame_out="yumi_body")
    return util.convert_pose_type(pose_array,
                                              type_out="PoseStamped",
                                              frame_out="yumi_body")

def compute_fk_dual(joints_list):
    pose_list = []
    for counter, joints in enumerate(joints_list):
        if counter==1:
            matrix = fk_solver_l.forward(joints)
        else:
            matrix = fk_solver_r.forward(joints)
        translation = tfm.translation_from_matrix(matrix)
        quat = tfm.quaternion_from_matrix(matrix)
        pose_list.append(np.hstack((translation, quat)))
    return pose_list

def compute_cost(joints, seed, tol = 5*np.pi/180):
    dif = np.array(joints) - np.array(seed)
    crazy_cost = 0
    # for n in range(len(dif)):
    #     if abs(dif[n]) > tol:
    #         crazy_cost += 10000
            # print ('COmpute IK: ', 'CRAZY**********************8')
            # rospy.set_param('is_feasible', False)
    cost = np.dot(dif, dif)
    return cost

def compute_ik(end_effector_pose, seed, arm="r"):
    joint_list = []
    cost_list = []

    for counter in range(15):
        seed_state = seed
        if arm=="r":
            joints = ik_solver_r.get_ik(seed_state,
                                      end_effector_pose[0],
                                      end_effector_pose[1],
                                      end_effector_pose[2],
                                      end_effector_pose[3],
                                      end_effector_pose[4],
                                      end_effector_pose[5],
                                      end_effector_pose[6])
        else:
            joints = ik_solver_l.get_ik(seed_state,
                                      end_effector_pose[0],
                                      end_effector_pose[1],
                                      end_effector_pose[2],
                                      end_effector_pose[3],
                                      end_effector_pose[4],
                                      end_effector_pose[5],
                                      end_effector_pose[6])
        try:
            if compute_cost(list(joints), seed) < 2.5:
                return list(joints)
            # joint_list.append(list(joints))
            # cost_list.append(compute_cost(list(joints), seed))
        except:
            continue
    return []

    # index = np.argmin(np.array(cost_list))
    #
    # return joint_list[index]

def compute_moveit_ik(end_effector_pose, seed, group="right_arm"):
    req = GetPositionIKRequest()
    req.ik_request.group_name = group
    arm = "l" if group == "left_arm" else "r"
    if group == "left_arm":
        req.ik_request.robot_state.joint_state.name = ['yumi_joint_%d_l' % i for i in [1, 2, 7, 3, 4, 5, 6]]
    else:
        req.ik_request.robot_state.joint_state.name = ['yumi_joint_%d_r' % i for i in [1, 2, 7, 3, 4, 5, 6]]
    req.ik_request.robot_state.joint_state.position = seed
    req.ik_request.pose_stamped = end_effector_pose
    req.ik_request.timeout = rospy.Duration(0.05)
    req.ik_request.attempts = 20
    res = ik_srv(req).solution
    return res.joint_state.position
