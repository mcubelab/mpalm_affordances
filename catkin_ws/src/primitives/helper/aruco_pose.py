import os, os.path as osp
import sys
import numpy as np
import threading
import copy
import rospy
import time
from geometry_msgs.msg import PoseStamped, TransformStamped

from airobot.utils import common
import util2 as util


class ArucoPose():
	def __init__(self):
		self.pose_matrix = np.eye(4)
		self.pose_sub = rospy.Subscriber('/aruco_single/pose', PoseStamped, self.pose_callback)
		self.pose_lock = threading.RLock()

	def pose_callback(self, data):
		pose_list = util.pose_stamped2list(data)
		R = common.quat2rot(pose_list[3:])
		t = pose_list[:3]
		self.pose_matrix[:-1, :-1] = R
		self.pose_matrix[:-1, -1] = t

	def get_current_pose(self):
		self.pose_lock.acquire()
		pose_matrix = copy.deepcopy(self.pose_matrix)
		self.pose_lock.release()
		return pose_matrix

def main():
	aruco = ArucoPose()
	rospy.init_node('PoseGetter')

	time.sleep(1.0)

	target_pose = util.pose_from_matrix(aruco.get_current_pose())
	print('Current Pose: ')
	print(target_pose)

	input('Waiting for you to move to start pose...')

	start_pose = util.pose_from_matrix(aruco.get_current_pose())
	print('Current Pose: ')
	print(start_pose)

	print('Computing relative transformation from start to goal')
	transformation_des = util.matrix_from_pose(util.get_transform(target_pose, start_pose))

	from IPython import embed
	embed()

if __name__ == '__main__':
	main()