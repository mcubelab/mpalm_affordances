import rospy
import tf

from rpo_planning.utils import common as util

def simulate(grasp_plan, object_name="realsense_box_experiments.stl"):
    import time, os, sys

    for plan_dict in grasp_plan:
        for i, t in enumerate(plan_dict['t']):
            visualize_object_pose(plan_dict['object_poses_world'][i], object_name=object_name)
            update_yumi_cart(plan_dict['palm_poses_world'][i])
            time.sleep(.1)

def simulate_palms(grasp_plan, object_name="realsense_box_experiments.stl"):
    import time, os, sys

    for plan_dict in grasp_plan:
        for i, t in enumerate(plan_dict['t']):
            update_yumi_cart(plan_dict['palm_poses_world'][i])
            time.sleep(.1)            

import rospy
from visualization_msgs.msg import Marker


# def visualize_object_pose(q, object_name="realsense_box_experiments.stl"):
#     for i in range(4):
#         visualize_object(q,
#                          filepath="package://config/descriptions/meshes/objects/" + object_name,
#                          name="/object",
#                          color=(1.0, 126.0 / 255.0, 34.0 / 255.0, 1.),
#                          frame_id="/yumi_body")
def visualize_object_pose(q, object_name):
    for i in range(4):
        visualize_object(q,
                         filepath="package://config/descriptions/meshes/objects/cuboids/" + object_name,
                         name="/object",
                         color=(1.0, 126.0 / 255.0, 34.0 / 255.0, 1.),
                         frame_id="/yumi_body")


def update_yumi_cart(poses):
    wrist_to_tip = util.list2pose_stamped([0.0, 0.071399, -0.14344421, 0.0, 0.0, 0.0, 1.0], '')
    world_to_world = util.unit_pose()
    wrist_left = util.convert_reference_frame(wrist_to_tip, world_to_world, poses[0], "yumi_body")
    wrist_right = util.convert_reference_frame(wrist_to_tip, world_to_world, poses[1], "yumi_body")
    visualize_object(wrist_left,
                     filepath="package://config/descriptions/meshes/mpalm/mpalms_all_coarse.stl",
                     name="/gripper_left",
                     color=(0., 1., 0., 1.),
                     frame_id="/yumi_body")

    visualize_object(wrist_right,
                     filepath="package://config/descriptions/meshes/mpalm/mpalms_all_coarse.stl",
                     name="/gripper_right",
                     color=(0., 0., 1., 1.),
                     frame_id="/yumi_body")


def visualize_object(pose, filepath="package://config/descriptions/meshes/objects/object.stl", name="/object",
                     color=(0., 0., 1., 1.), frame_id="/yumi_body", scale=(1., 1., 1.)):
    marker_pub = rospy.Publisher(name, Marker, queue_size=1)
    marker_type = Marker.MESH_RESOURCE
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = name
    marker.header.stamp = rospy.Time(0)
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.type = marker_type
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.lifetime.secs = 1
    marker.pose = pose.pose
    marker.mesh_resource = filepath
    marker.lifetime = rospy.Duration(10000)

    for i in range(10):
        marker_pub.publish(marker)


def handle_block_pose(msg, br, base_frame, target_frame):
    """visualize reference frame in rviz"""
    for i in range(3):
        br.sendTransform((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                         (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w),
                         rospy.Time.now(),
                         target_frame,
                         base_frame)