import os, os.path as osp
import numpy as np

import sys
import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
from rpo_lcm import point_t, quaternion_t, pose_t, pose_stamped_t, point_cloud_t, skill_param_t
from rpo_planning.lcm_types.rpo_lcm import point_cloud_t, skill_param_t

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