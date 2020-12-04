#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
# ~ from real.camera import Camera
import sys
import os, os.path as osp
sys.path.append(osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives'))
from scipy import optimize  
from mpl_toolkits.mplot3d import Axes3D  
from sensor_msgs.msg import Image
import copy

import rospy
import std_srvs.srv
import sensor_msgs.msg
import geometry_msgs.msg._Transform as transform

from cv_bridge import CvBridge
import threading
import argparse

from airobot import Robot


class RealsenseCalibration(object):
	def __init__(self, camera_name):
		self.camera_name = camera_name
		# rgb_topic = '/' + camera_name + '/color/image_rect_color'
		depth_topic = '/' + camera_name + '/depth/image_rect_raw'

		rgb_topic = '/' + camera_name + '/color/image_raw'
		print('TOPICS: ', rgb_topic, depth_topic)
		self.rs_rgb_sub = rospy.Subscriber(rgb_topic, Image, self.rgb_cb)
		self.rs_depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_cb)     

		self._rgb_lock = threading.RLock()
		self._depth_lock = threading.RLock()

		self._rgb_image = None
		self._depth_image = None
		self._depth_scale = 0.001

		self._cv_bridge = CvBridge()

	def rgb_cb(self, data):
		self._rgb_image = data

	def depth_cb(self, data):
		self._depth_image = data

	def getFrame(self):
		self._rgb_lock.acquire()
		img = copy.deepcopy(self._rgb_image)
		self._rgb_lock.release()
		return img

	def get_image(self, topic):
		image_msg = rospy.wait_for_message(topic, Image)
		outImage = self._cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
		return outImage 

	def get_depth(self):
		self._depth_lock.acquire()
		depth_msg = copy.deepcopy(self._depth_image)
		self._depth_lock.release()
		out_depth = self._cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough') * self._depth_scale
		# out_depth = np.round(depth_msg * 10000).astype(np.uint16)
		return out_depth


# Estimate rigid transform with SVD (from Nghia Ho)
def get_rigid_transform(A, B):
	assert len(A) == len(B)
	N = A.shape[0]; # Total points
	centroid_A = np.mean(A, axis=0)
	centroid_B = np.mean(B, axis=0)
	AA = A - np.tile(centroid_A, (N, 1)) # Centre the points
	BB = B - np.tile(centroid_B, (N, 1))
	H = np.dot(np.transpose(AA), BB) # Dot is matrix multiplication for array
	U, S, Vt = np.linalg.svd(H)
	R = np.dot(Vt.T, U.T)
	if np.linalg.det(R) < 0: # Special reflection case
	   Vt[2,:] *= -1
	   R = np.dot(Vt.T, U.T)
	t = np.dot(-R, centroid_A.T) + centroid_B.T
	return R, t

def get_rigid_transform_error(z_scale):
	global measured_pts, observed_pts, observed_pix, world2camera, camera, cam_intrinsics

	# Apply z offset and compute new observed points using camera intrinsics
	observed_z = observed_pts[:,2:] * z_scale
	observed_x = np.multiply(observed_pix[:,[0]]-cam_intrinsics[0][2],observed_z/cam_intrinsics[0][0])
	observed_y = np.multiply(observed_pix[:,[1]]-cam_intrinsics[1][2],observed_z/cam_intrinsics[1][1])
	new_observed_pts = np.concatenate((observed_x, observed_y, observed_z), axis=1)
	print("new_observed_pts",new_observed_pts)
	print("measured_pts",measured_pts)
	
	# Estimate rigid transform between measured points and new observed points
	R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(new_observed_pts))
	t.shape = (3,1)
	world2camera = np.concatenate((np.concatenate((R, t), axis=1),np.array([[0, 0, 0, 1]])), axis=0)

	# Compute rigid transform error
	registered_pts = np.dot(R,np.transpose(measured_pts)) + np.tile(t,(1,measured_pts.shape[0]))
	error = np.transpose(registered_pts) - new_observed_pts
	error = np.sum(np.multiply(error,error))
	rmse = np.sqrt(error/measured_pts.shape[0]);
	return rmse


def main(args):
	global measured_pts, observed_pts, observed_pix, world2camera, camera, cam_intrinsics

	with_robot = args.robot
	calib = RealsenseCalibration(camera_name=args.cam)
	getFrame = calib.getFrame
	get_image = calib.get_image
	save_dir = osp.join(os.getcwd(), 'calibration', args.cam)
	if not osp.exists(save_dir):
		os.makedirs(save_dir)

	cam_1_intrinsics = np.asarray([
		[616.622, 0, 304.482], 
		[0, 616.122, 234.00], 
		[0, 0, 1]])

	cam_2_intrinsics = np.asarray([
		[619.131, 0, 331.664], 
		[0, 618.779, 237.653], 
		[0, 0, 1]])

	cam_intrinsics = cam_1_intrinsics if args.cam == 'cam_1' else cam_2_intrinsics  

	rospy.init_node('calibration_realsense')
	
	# User options (change me)
	# --------------- Setup options ---------------
	workspace_limits = np.asarray([[325, 475], [-350, -100], [150, 200]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates) # SET THIS
	workspace_limits = workspace_limits / 1000.0 # change to m
	# ~ calib_grid_step = 50
	calib_grid_num = [4,5,2]
	checkerboard_offset_from_tool = np.array([0.1, 0 ,0])
	tool_orientation = [0.5, -0.5, 0.5, -0.5] # SET THIS
	# tool_orientation = [0.0, 0.7071, 0.0, 0.70710]
	# tool_orientation = [0.7071, 0.0, 0.70710, 0.0]
	home_pose_joints=[67.85, -124.72, -95.94, 30.75, 71.48, 43.58, 29.32] # SET THIS (new -- airobot convenstion, with 1,2,7...
	# ---------------------------------------------


	# Construct 3D calibration grid across workspace
	# ~ gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], 1 + (workspace_limits[0][1] - workspace_limits[0][0])/calib_grid_step)
	gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], num=calib_grid_num[0])
	# ~ gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], 1 + (workspace_limits[1][1] - workspace_limits[1][0])/calib_grid_step)
	gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], num=calib_grid_num[1])
	# ~ gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], 1 + (workspace_limits[2][1] - workspace_limits[2][0])/calib_grid_step)
	gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], num=calib_grid_num[2])

	calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
	num_calib_grid_pts = calib_grid_x.shape[0]*calib_grid_x.shape[1]*calib_grid_x.shape[2]
	calib_grid_x.shape = (num_calib_grid_pts,1)
	calib_grid_y.shape = (num_calib_grid_pts,1)
	calib_grid_z.shape = (num_calib_grid_pts,1)
	calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)

	measured_pts = []
	observed_pts = []
	observed_pix = []



	if with_robot: 
		# Move robot to home pose
		print('Connecting to robot...')
		# yumi_robot = yumi_helper.Yumi()
		yumi_robot = Robot('yumi_palms', pb=False, use_cam=False)

		# Slow down robot
		yumi_robot.arm.right_arm.set_speed(100, 50) # TODO!!!!

		# Make robot to home pose
		yumi_robot.arm.right_arm.set_jpos(np.deg2rad(home_pose_joints), wait=False)

		# Move robot to each calibration point in workspace
		print('Collecting data...')
		newdata = []
		for calib_pt_idx in range(num_calib_grid_pts):
			robot_position = calib_grid_pts[calib_pt_idx, :]
			# tool_position = calib_grid_pts[calib_pt_idx,:]
			tool_position = robot_position + checkerboard_offset_from_tool
			# print 'tool_position', tool_position
			if not with_robot: pass
			else:
				
				yumi_robot.arm.right_arm.set_ee_pose(
					[robot_position[0], robot_position[1], robot_position[2]], 
					[tool_orientation[0], tool_orientation[1], tool_orientation[2], tool_orientation[3]],
					wait=False)
				time.sleep(1)
			
				#################################################################################################################
				
				#BECAUSE THE FRAMES ARE BEING PUBLISHED LIVE IN A SEPERATE SCRIPT - WE SHOULD PAUSE THIS SCRIPT AND WAIT FOR THE ROBOT TO MOVE TO THE DESIRED POSITION BEFORE READING IN THE PICTURE 
				time.sleep(3) # can change 
				
				# Find checkerboard center
				checkerboard_size = (3,3)
				refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

				#camera_color_img, camera_depth_img = robot.get_camera_data()
				# GET GRAYSCALE AND DEPTH IMAGE 
				# print("GET GRAYSCALE AND DEPTH IMAGES")

				# run the photoneo publish script
				
				# get rgb image 
				texture_np = get_image('/' + args.cam + '/color/image_raw')
				texture_np = cv2.cvtColor(texture_np, cv2.COLOR_BGR2GRAY)
				# print("RECEIVED RGB DATA")

				# obtain the depth from the pointcloud file
				# depth = texture_np *0
				depth = calib.get_depth()

				# normalize the grayscale image to be brighter
				# print("NORMALIZE THE GRAYSCALE IMAGE")
				max_grayscale = float(np.amax(texture_np))
				norm_texture_np = np.asarray((texture_np/max_grayscale)*255, dtype=np.uint8)

				# plt.imshow(norm_texture_np.astype(np.float32),cmap='gray')
				# plt.show()

				# plt.imshow(depth)
				# plt.show()
				# convert to correct syntax that the original code used
				camera_depth_img = depth
				camera_color_img = texture_np # instead of rgb -  use grayscale
				gray_data_2 = texture_np
				#gray_data_2 = (gray_data/256).astype('uint8')
				
				
				###################################################
				
				# ~ # Find checkerboard center
				# ~ checkerboard_size = (3,3)
				# ~ refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
				# ~ camera_color_img, camera_depth_img = robot.get_camera_data()
				# ~ bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
				# ~ gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)
				
				checkerboard_found, corners = cv2.findChessboardCorners(gray_data_2, checkerboard_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
				# print("FIND CHECKERBOARD ", checkerboard_found)
				if checkerboard_found:
					corners_refined = cv2.cornerSubPix(gray_data_2, corners, (3,3), (-1,-1), refine_criteria)

					# Get observed checkerboard center 3D point in camera space
					checkerboard_pix = np.round(corners_refined[4,0,:]).astype(int)
					checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
					checkerboard_x = np.multiply(checkerboard_pix[0]-cam_intrinsics[0][2],checkerboard_z/cam_intrinsics[0][0])
					checkerboard_y = np.multiply(checkerboard_pix[1]-cam_intrinsics[1][2],checkerboard_z/cam_intrinsics[1][1])
					print('checkerboard [x, y, z]: %.3f, %.3f, %.3f' % (checkerboard_x, checkerboard_y, checkerboard_z))
					if np.abs(checkerboard_z) < 1e-4:
						print('skipping')
						continue

					# Save calibration point and observed checkerboard center
					observed_pts.append([checkerboard_x,checkerboard_y,checkerboard_z])

					measured_pts.append(tool_position)
					observed_pix.append(checkerboard_pix)
					
					# Draw and display the corners
					# vis = cv2.drawChessboardCorners(robot.camera.color_data, checkerboard_size, corners_refined, checkerboard_found)
					vis = cv2.drawChessboardCorners(gray_data_2, (1,1), corners_refined[4,:,:], checkerboard_found)
					cv2.imwrite(osp.join(save_dir, '%06dd.png' % len(measured_pts)), vis)
					# cv2.imshow('Calibration',vis)
					# cv2.waitKey(0)

					# print(tool_position, checkerboard_pix)
					newdata.append({"cross3d": tool_position.tolist(), "pic_path": '%06dd.png' % len(measured_pts), "cross2d":checkerboard_pix.tolist()})
					
					import json
					with open(osp.join(save_dir, 'data.extracted2d.json'), 'w') as outfile:
						json.dump(newdata, outfile)
		# show_updated(tuple(corners.tolist()[0]))

		# while True:
			#display the image and wait for a keypress
			# key = cv2.waitKey(3) & 0xFF
			# if key == ord("n"):
				# break

	if with_robot: 
		# Move robot back to home pose
		yumi_robot.arm.right_arm.set_ee_pose(
			[robot_position[0], robot_position[1], robot_position[2] + 0.150], 
			[tool_orientation[0], tool_orientation[1], tool_orientation[2], tool_orientation[3]],
			wait=False)
		yumi_robot.arm.right_arm.set_jpos(np.deg2rad(home_pose_joints), wait=False)

		measured_pts = np.asarray(measured_pts)
		observed_pts = np.asarray(observed_pts)
		observed_pix = np.asarray(observed_pix)

		try:
			np.save(osp.join(save_dir, 'measured_pts'), measured_pts)
			np.save(osp.join(save_dir, 'observed_pts'), observed_pts)
			np.save(osp.join(save_dir, 'observed_pix'), observed_pix)
		except IOError as e:
			print(e)
			from IPython import embed
			embed()

	else:
		measured_pts = np.load(osp.join(save_dir, 'measured_pts.npy'))
		observed_pts = np.load(osp.join(save_dir, 'observed_pts.npy'))
		observed_pix = np.load(osp.join(save_dir, 'observed_pix.npy') ) 

	world2camera = np.eye(4)

	# Optimize z scale w.r.t. rigid transform error
	print('Calibrating...')

	z_scale_init = 1
	optim_result = optimize.minimize(get_rigid_transform_error, np.asarray(z_scale_init), method='Nelder-Mead')
	camera_depth_offset = optim_result.x

	# Save camera optimized offset and camera pose
	print('Saving...')
	np.savetxt(osp.join(save_dir, 'camera_depth_scale.txt'), camera_depth_offset, delimiter=' ')
	get_rigid_transform_error(camera_depth_offset)
	camera_pose = np.linalg.inv(world2camera)
	np.savetxt(osp.join(save_dir, 'camera_pose.txt'), camera_pose, delimiter=' ')
	print('Done.')

	# DEBUG CODE -----------------------------------------------------------------------------------

	np.savetxt(osp.join(save_dir, 'measured_pts.txt'), np.asarray(measured_pts), delimiter=' ')
	np.savetxt(osp.join(save_dir, 'observed_pts.txt'), np.asarray(observed_pts), delimiter=' ')
	np.savetxt(osp.join(save_dir, 'observed_pix.txt'), np.asarray(observed_pix), delimiter=' ')
	measured_pts = np.loadtxt(osp.join(save_dir, 'measured_pts.txt'), delimiter=' ')
	observed_pts = np.loadtxt(osp.join(save_dir, 'observed_pts.txt'), delimiter=' ')
	observed_pix = np.loadtxt(osp.join(save_dir, 'observed_pix.txt'), delimiter=' ')

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(measured_pts[:,0],measured_pts[:,1],measured_pts[:,2], c='blue')

	print(camera_depth_offset)
	R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(observed_pts))
	t.shape = (3,1)
	camera_pose = np.concatenate((np.concatenate((R, t), axis=1),np.array([[0, 0, 0, 1]])), axis=0)
	camera2robot = np.linalg.inv(camera_pose)
	t_observed_pts = np.transpose(np.dot(camera2robot[0:3,0:3],np.transpose(observed_pts)) + np.tile(camera2robot[0:3,3:],(1,observed_pts.shape[0])))

	ax.scatter(t_observed_pts[:,0],t_observed_pts[:,1],t_observed_pts[:,2], c='red')

	new_observed_pts = observed_pts.copy()
	new_observed_pts[:,2] = new_observed_pts[:,2] * camera_depth_offset[0]
	R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(new_observed_pts))
	t.shape = (3,1)
	camera_pose = np.concatenate((np.concatenate((R, t), axis=1),np.array([[0, 0, 0, 1]])), axis=0)
	camera2robot = np.linalg.inv(camera_pose)
	t_new_observed_pts = np.transpose(np.dot(camera2robot[0:3,0:3],np.transpose(new_observed_pts)) + np.tile(camera2robot[0:3,3:],(1,new_observed_pts.shape[0])))

	ax.scatter(t_new_observed_pts[:,0],t_new_observed_pts[:,1],t_new_observed_pts[:,2], c='green')

	plt.show()

	from airobot.utils import common
	print(camera2robot[:-1, -1].tolist() + common.rot2quat(camera2robot[:-1, :-1]).tolist())

	trans = camera2robot[:-1, -1].tolist()
	quat = common.rot2quat(camera2robot[:-1, :-1]).tolist()

	ret = {
		'b_c_transform': {
			'position': trans,
			'orientation': quat,
			'T': camera2robot.tolist()
		}
	}

	calib_file_dir = os.path.join(os.environ['CODE_BASE'], args.data_path)
	if not osp.exists(calib_file_dir):
		os.makedirs(calib_file_dir)
	calib_file_path = osp.join(calib_file_dir, args.cam + '_calib_base_to_cam.json')
	print(json.dumps(ret, indent=2))
	with open(calib_file_path, 'w') as fp:
		json.dump(ret, fp, indent=2)    


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--cam', type=str, default='cam_1')
	parser.add_argument('--robot', action='store_true')
	parser.add_argument('--data_path', type=str, default='catkin_ws/src/hand_eye_calibration/result/yumi')  
	args = parser.parse_args()
	main(args)