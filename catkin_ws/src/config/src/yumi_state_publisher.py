#!/usr/bin/env python

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from robot_comm.msg import robot_JointsLog

# rospy.set_param('have_robot', True)

def callback_robot_1(msg, joint_states_pub):
    if rospy.get_param('have_robot'):
        jnames = ['yumi_joint_1_r',
                  'yumi_joint_2_r',
                  'yumi_joint_3_r',
                  'yumi_joint_4_r',
                  'yumi_joint_5_r',
                  'yumi_joint_6_r',
                  'yumi_joint_7_r']

        js = JointState()
        js.name = jnames
        js.position = [val/180.0*np.pi for val in msg.position]
        joint_states_pub.publish(js)

def callback_robot_2(msg, joint_states_pub):
    if rospy.get_param('have_robot'):
        jnames = ['yumi_joint_1_l',
                  'yumi_joint_2_l',
                  'yumi_joint_3_l',
                  'yumi_joint_4_l',
                  'yumi_joint_5_l',
                  'yumi_joint_6_l',
                  'yumi_joint_7_l']

        js = JointState()
        js.name = jnames
        js.position = [val/180.0*np.pi for val in msg.position]
        joint_states_pub.publish(js)

def callback_joint_state(msg):
    joint_states = msg

if __name__ == '__main__':
    rospy.init_node('yumi_state_publisher', anonymous=True)

    joint_states_pub = rospy.Publisher("/yumi/joint_states", JointState, queue_size=10)
    robot1_joints_sub = rospy.Subscriber("/robot1_JointState", JointState, callback_robot_1, joint_states_pub)
    robot2_joints_sub = rospy.Subscriber("/robot2_JointState", JointState, callback_robot_2, joint_states_pub)
    rospy.spin()
