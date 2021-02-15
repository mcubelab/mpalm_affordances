import os, os.path as osp
import numpy as np

import sys
import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
from rpo_lcm import (
    point_t, quaternion_t, pose_t, pose_stamped_t, 
    point_cloud_t, skill_param_t, dual_pose_stamped_t)
from rpo_planning.lcm_types.rpo_lcm import point_cloud_t, skill_param_t
from rpo_planning.utils import common as util

def np2point_cloud_t(pcd_np, frame_id='world'):
    pcd_t = point_cloud_t()
    for i in range(pcd_np.shape[0]):
        pt_msg = point_t()
        pt_msg.x = pcd_np[i, 0]
        pt_msg.y = pcd_np[i, 1]
        pt_msg.z = pcd_np[i, 2]
        pcd_t.points.append(pt_msg)
    pcd_t.num_points = pcd_np.shape[0]
    pcd_t.header.frame_name = frame_id
    return pcd_t

def pose_stamped2list(msg):
    return [float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
            ]

def list2pose_stamped_lcm(pose, frame_id='world'):
    ps_t = pose_stamped_t()
    ps_t.header.frame_name = frame_id
    ps_t.pose.position.x = pose[0]
    ps_t.pose.position.y = pose[1]
    ps_t.pose.position.z = pose[2]
    ps_t.pose.orientation.x = pose[3]
    ps_t.pose.orientation.y = pose[4]
    ps_t.pose.orientation.z = pose[5]
    ps_t.pose.orientation.w = pose[6]
    return ps_t 

def matrix2pose_stamped_lcm(pose_mat, frame_id='world'):
    pose_msg = util.pose_from_matrix(pose_mat)
    pose_list = util.pose_stamped2list(pose_msg)
    return list2pose_stamped_lcm(pose_list) 

def fill_skill_subgoal_and_contact(contact_pose, subgoal_pose, sp_t):
    """
    Function to fill in the contact pose and subgoal pose data into an
    RPO skill_param_t message. Assumes that ALL OTHER FIELDS in sp_t have
    been filled (so that each point in the point cloud field can be associated
    with a contact pose)

    Args:
        contact_pose (np.ndarray): Either single arm or dual arm
            contact pose (TODO: determine how to handle either R-single or L-single arm)
        subgoal_pose (np.ndarray): Subgoal relative transformation, expressed as 4 x 4 HTM
        sp_t (skill_param_t): LCM message to be filled
    """
    assert sp_t.num_points > 0, 'num_points field must be filled with a value greater than 0' 

    sp_t.subgoal_pose = matrix2pose_stamped_lcm(subgoal_pose)
    for j in range(sp_t.num_points):
        tc_r = list2pose_stamped_lcm(contact_pose[:7])
        if contact_pose.shape[0] > 7: 
            tc_l = list2pose_stamped_lcm(contact_pose[7:])
        else:
            tc_l = tc_r 
        tc = dual_pose_stamped_t()
        tc.right_pose = tc_r
        tc.left_pose = tc_l
        sp_t.contact_pose.append(tc)
    
def unpack_pointcloud_lcm(points, num_points):
    """
    Function to unpack a point cloud LCM message into a list

    Args:
        points (list): Each element is point_cloud_t type, with fields
            x, y, z
        num_points (int): Number of points in the point cloud
    """
    pt_list = []
    for i in range(num_points):
        pt = [
            points[i].x,
            points[i].y,
            points[i].z
        ]
        pt_list.append(pt)
    return pt_list
