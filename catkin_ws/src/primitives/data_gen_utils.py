import os, os.path as osp
import sys
import time
import argparse
import numpy as np
from multiprocessing import Process, Pipe, Queue
import pickle
import rospy
import copy
import signal
import open3d
import threading
import cv2
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
from IPython import embed

from airobot import Robot
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from airobot.sensor.camera.rgbdcam_real import RGBDCameraReal
from airobot.utils.ros_util import read_cam_ext
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions
from yumi_pybullet_ros import YumiGelslimPybullet
from yumi_real_ros import YumiGelslimReal

from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

from yacs.config import CfgNode as CN
from closed_loop_experiments_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler


class YumiCamsGS(YumiGelslimPybullet):
    """
    Child class of YumiGelslimPybullet with additional functions
    for setting up multiple cameras in the pybullet scene
    and getting observations of various types
    """
    def __init__(self, yumi_pb, cfg, exec_thread=True, sim_step_repeat=10):
        """
        Constructor, sets up base class and additional camera setup
        configuration parameters.

        Args:
            yumi_pb (airobot Robot): Instance of PyBullet simulated robot, from
                airobot library
            cfg (YACS CfgNode): Configuration parameters
            exec_thread (bool, optional): Whether or not to start the
                background joint position control thread. Defaults to True.
            sim_step_repeat (int, optional): Number of simulation steps
                to take each time the desired joint position value is
                updated. Defaults to 10
        """
        super(YumiCamsGS, self).__init__(yumi_pb,
                                         cfg,
                                         exec_thread=exec_thread,
                                         sim_step_repeat=sim_step_repeat)

        self.cams = []
        for _ in range(4):
            self.cams.append(RGBDCameraPybullet(cfgs=self._camera_cfgs(),
                                                pb_client=yumi_pb.pb_client))

        self.cam_setup_cfg = {}
        # self.cam_setup_cfg['focus_pt'] = [self.cfg.OBJECT_POSE_3[:3]]*3
        # self.cam_setup_cfg['dist'] = [0.7, 0.7, 0.75]
        # self.cam_setup_cfg['yaw'] = [30, 150, 270]
        # self.cam_setup_cfg['pitch'] = [-45, -45, -70]
        # self.cam_setup_cfg['roll'] = [0, 0, 0]
        self.cam_setup_cfg['focus_pt'] = [self.cfg.CAMERA_FOCUS]*4
        self.cam_setup_cfg['dist'] = [0.8, 0.8, 0.8, 0.8]
        self.cam_setup_cfg['yaw'] = [30, 150, 210, 330]
        self.cam_setup_cfg['pitch'] = [-35, -35, -20, -20]
        self.cam_setup_cfg['roll'] = [0, 0, 0, 0]

        self._setup_cameras()

    def _camera_cfgs(self):
        """
        Returns a set of camera config parameters

        Returns:
            YACS CfgNode: Cam config params
        """
        _C = CN()
        _C.ZNEAR = 0.01
        _C.ZFAR = 10
        _C.WIDTH = 640
        _C.HEIGHT = 480
        _C.FOV = 60
        _ROOT_C = CN()
        _ROOT_C.CAM = CN()
        _ROOT_C.CAM.SIM = _C
        return _ROOT_C.clone()

    def _setup_cameras(self):
        """
        Function to set up 3 pybullet cameras in the simulated environment
        """
        self.cams[0].setup_camera(
            focus_pt=self.cam_setup_cfg['focus_pt'][0],
            dist=self.cam_setup_cfg['dist'][0],
            yaw=self.cam_setup_cfg['yaw'][0],
            pitch=self.cam_setup_cfg['pitch'][0],
            roll=self.cam_setup_cfg['roll'][0]
        )
        self.cams[1].setup_camera(
            focus_pt=self.cam_setup_cfg['focus_pt'][1],
            dist=self.cam_setup_cfg['dist'][1],
            yaw=self.cam_setup_cfg['yaw'][1],
            pitch=self.cam_setup_cfg['pitch'][1],
            roll=self.cam_setup_cfg['roll'][1]
        )
        self.cams[2].setup_camera(
            focus_pt=self.cam_setup_cfg['focus_pt'][2],
            dist=self.cam_setup_cfg['dist'][2],
            yaw=self.cam_setup_cfg['yaw'][2],
            pitch=self.cam_setup_cfg['pitch'][2],
            roll=self.cam_setup_cfg['roll'][2]
        )
        self.cams[3].setup_camera(
            focus_pt=self.cam_setup_cfg['focus_pt'][3],
            dist=self.cam_setup_cfg['dist'][3],
            yaw=self.cam_setup_cfg['yaw'][3],
            pitch=self.cam_setup_cfg['pitch'][3],
            roll=self.cam_setup_cfg['roll'][3]
        )

    def get_observation(self, obj_id, depth_max=1.0, 
                        downsampled_pcd_size=100, robot_table_id=None,
                        cam_inds=None, depth_noise=False,
                        depth_noise_std=0.0025, depth_noise_rate=0.00025):
        """
        Function to get an observation from the pybullet scene. Gets
        an RGB-D images and point cloud from each camera viewpoint,
        along with a segmentation mask using PyBullet's builtin
        segmentation mask functionality. Uses the segmentation mask
        to build a segmented point cloud

        Args:
            obj_id (int): PyBullet object id, used to compute segmentation
                mask.
            depth_max (float, optional): Max depth to capture in depth image.
                Defaults to 1.0.
            downsampled_pcd_size (int, optional): Number of points to downsample
                the pointcloud to
            robot_table_id (tuple): Tuple of size 2 with (robot_id, link_id) of the
                table, or whatever target surface is desired to obtain the pointcloud
                of
            cam_inds (list): List of integer values, which indicate the indices of the
                camera images we want to include in the fused point cloud scene, i.e.
                [0, 1, 3] would cause us to leave out the image obtained from camera
                with index 2 in self.cams
            depth_noise (bool): True if we should add noise to the obtained depth image
            depth_noise_std (float): Baseline standard deviation of Gaussian noise to add to the
                point cloud, which is scaled up based on z-distance away from the depth image            
            depth_noise_rate (float): Coefficient of the depth**2 term in the noise model

        Returns:
            2-element tuple containing:
                - dict: Contains observation data, with keys for
                    - rgb: list of np.ndarrays for each RGB image
                    - depth: list of np.ndarrays for each depth image
                    - seg: list of np.ndarrays for each segmentation mask
                    - pcd_pts: list of np.ndarrays for each segmented point cloud
                    - pcd_colors: list of np.ndarrays for the colors corresponding
                        to the points of each segmented pointcloud
                    - table_pcd_pts: list of np.ndarrays for table pointcloud from different
                        views
                    - table_pcd_colors: list of np.ndarrays for colors corresponding
                        to the points of each segmented table pointcloud
                    - down_pcd_pts: downsampled version of the object pointcloud
                    - down_pcd_normals: estimated normals at each point in the downsampled pointcloud
                    - center_of_mass: estimated center of mass of the object point cloud
                - open3d.geometry.PointCloud: Contains object point cloud in open3d format
        """
        rgbs = []
        depths = []
        segs = []
        obj_pcd_pts = []
        obj_pcd_colors = []
        table_pcd_pts = []
        table_pcd_colors = []

        for i, cam in enumerate(self.cams):
            # skip cam inds that are not specified
            if cam_inds is not None:
                if i not in cam_inds:
                    continue
            rgb, depth, seg = cam.get_images(
                get_rgb=True,
                get_depth=True,
                get_seg=True
            )

            if depth_noise:
                depth = self.apply_noise_to_seg_depth(depth, seg, obj_id, std=depth_noise_std, rate=depth_noise_rate)

                pts_raw, colors_raw = cam.get_pcd(
                    in_world=True,
                    filter_depth=False,
                    depth_max=depth_max,
                    force_rgb=rgb,
                    force_depth=depth
                )
            else:
                pts_raw, colors_raw = cam.get_pcd(
                    in_world=True,
                    filter_depth=False,
                    depth_max=depth_max
                )                


            flat_seg = seg.flatten()
            obj_inds = np.where(flat_seg == obj_id)
            obj_pts, obj_colors = pts_raw[obj_inds[0], :], colors_raw[obj_inds[0], :]

            rgbs.append(copy.deepcopy(rgb))
            depths.append(copy.deepcopy(depth))
            segs.append(copy.deepcopy(seg))

            obj_pcd_pts.append(obj_pts)
            obj_pcd_colors.append(obj_colors)


            if robot_table_id is not None:
                if robot_table_id[1] == -1:
                    table_inds = np.where(flat_seg == robot_table_id[0])
                else:
                    robot_id, table_id = robot_table_id[0], robot_table_id[1]
                    table_val = robot_id + (table_id+1) << 24
                    table_inds = np.where(flat_seg == table_val)
                table_pts, table_colors = pts_raw[table_inds[0], :], colors_raw[table_inds[0], :]
                keep_inds = np.where(table_pts[:, 0] > 0.0)[0]
                table_pts = table_pts[keep_inds, :]
                table_colors = table_colors[keep_inds, :]
                table_pcd_pts.append(table_pts)
                table_pcd_colors.append(table_colors)

        pcd = open3d.geometry.PointCloud()

        obj_pcd_pts_cat = np.concatenate(obj_pcd_pts, axis=0)
        obj_pcd_colors_cat = np.concatenate(obj_pcd_colors, axis=0)

        pcd.points = open3d.utility.Vector3dVector(obj_pcd_pts_cat)
        pcd.colors = open3d.utility.Vector3dVector(obj_pcd_colors_cat / 255.0)
        pcd.estimate_normals()

        total_pts = obj_pcd_pts_cat.shape[0]
        down_pcd = pcd.uniform_down_sample(int(total_pts/downsampled_pcd_size))

        obs_dict = {}
        obs_dict['rgb'] = rgbs
        obs_dict['depth'] = depths
        obs_dict['seg'] = segs
        obs_dict['pcd_pts'] = obj_pcd_pts
        obs_dict['pcd_colors'] = obj_pcd_colors
        obs_dict['pcd_normals'] = np.asarray(pcd.normals)
        obs_dict['table_pcd_pts'] = table_pcd_pts
        obs_dict['table_pcd_colors'] = table_pcd_colors
        obs_dict['down_pcd_pts'] = np.asarray(down_pcd.points)
        obs_dict['down_pcd_normals'] = np.asarray(down_pcd.normals)
        obs_dict['center_of_mass'] = pcd.get_center()
        return obs_dict, pcd

    def apply_noise_to_seg_depth(self, depth, seg, obj_id, std, rate=0.00025):
        """Function to apply depth-dependent Gaussian noise to a depth
        image, with some baseline variance. Only points corresponding to the object
        in the depth image have noise added for computational efficiency. 

        Args:
            depth (np.ndarray): W x H x 1 np.ndarray of depth values from camera
            seg (np.ndarray): W x H x 1 np.ndarray of segmentation masks from simulated camera
            obj_id (int): PyBullet object id for object
            std (float): Baseline standard deviation to apply to depth values
            rate (float): Linear rate that standard deviation of noise should increase
                per meter**2 of depth value

        Returns:
            np.ndarray: W x H x 1 np.ndarray with same number of points as input, but with noise added
                to each value 
        """
        s = depth.shape
        flat_depth = depth.flatten()
        flat_seg = seg.flatten()
        obj_inds = np.where(flat_seg == obj_id)
        obj_depth = flat_depth[obj_inds[0]]
        eps = 0.0001

        new_depth = []
        for i in range(100):
            start, end = i*int(len(obj_depth)/100), (i+1)*int(len(obj_depth)/100)
            depth_window = obj_depth[start:end]
            std_dev = max(std + rate*np.mean(depth_window)**2, eps)
            noise_sample = np.random.normal(
                loc=0,
                scale=std_dev,
                size=depth_window.shape)
            new_depth_window = depth_window + noise_sample
            new_depth.append(new_depth_window)

        new_depth = np.asarray(new_depth).flatten()
        flat_depth[obj_inds[0][:len(new_depth)]] = new_depth

        return flat_depth.reshape(s)


class MultiRealsense(object):
    def __init__(self, n_cam=2, calib_fname_suffix='calib_base_to_cam.json'):
        self.cams = []
        self.names = []
        for i in range(1, n_cam+1):
            name = 'cam_%d' % i
            cam = RGBDCameraReal(cfgs=self._camera_cfgs(name), cam_name=name)

            # read_cam_ext obtains extrinsic calibration from file that has been previously saved
            pos, ori = read_cam_ext('yumi', name + '_' + calib_fname_suffix)
            cam.set_cam_ext(pos, ori)
            
            self.cams.append(cam)

    def _camera_cfgs(self, name):
        """
        Returns a set of camera config parameters
        Returns:
            YACS CfgNode: Cam config params
        """

        _C = CN()
        # topic name of the camera info
        _C.ROSTOPIC_CAMERA_INFO = '/color/camera_info'
        # topic name of the RGB images
        _C.ROSTOPIC_CAMERA_RGB = '/color/image_rect_color'
        # topic name of the depth images
        _C.ROSTOPIC_CAMERA_DEPTH = '/aligned_depth_to_color/image_raw'
        # minimum depth values to be considered as valid (m)
        _C.DEPTH_MIN = 0.2
        # maximum depth values to be considered as valid (m)
        _C.DEPTH_MAX = 2
        # scale factor to map depth image values to real depth values (m)
        _C.DEPTH_SCALE = 0.001

        _ROOT_C = CN()
        _ROOT_C.CAM = CN()
        _ROOT_C.CAM.REAL = _C

        return _ROOT_C.clone()


class YumiCamsGSReal(YumiGelslimReal):
    """
    Child class of YumiGelslimPybullet with additional functions
    for setting up multiple cameras in the pybullet scene
    and getting observations of various types
    """
    def __init__(self, yumi_ar, cfg, n_cam=4):
        """
        Constructor, sets up base class and additional camera setup
        configuration parameters.

        Args:
            yumi_ar (airobot Robot): AIRobot interface to real yumi
            cfg (YACS CfgNode): Configuration parameters
        """
        super(YumiCamsGSReal, self).__init__(yumi_ar, cfg)
        self.multicam_manager = MultiRealsense(n_cam=n_cam)
        self.cams = self.multicam_manager.cams
        self._setup_detectron()

    def get_observation(self, color_seg=False, depth_max=1.0, downsampled_pcd_size=100, robot_table_id=None):
        """
        Function to get an observation from the pybullet scene. Gets
        an RGB-D images and point cloud from each camera viewpoint,
        along with a segmentation mask using PyBullet's builtin
        segmentation mask functionality. Uses the segmentation mask
        to build a segmented point cloud

        Args:
            obj_id (int): PyBullet object id, used to compute segmentation
                mask.
            depth_max (float, optional): Max depth to capture in depth image.
                Defaults to 1.0.
            downsampled_pcd_size (int, optional): Number of points to downsample
                the pointcloud to
            robot_table_id (tuple): Tuple of size 2 with (robot_id, link_id) of the
                table, or whatever target surface is desired to obtain the pointcloud
                of

        Returns:
            dict: Contains observation data, with keys for
            - rgb: list of np.ndarrays for each RGB image
            - depth: list of np.ndarrays for each depth image
            - seg: list of np.ndarrays for each segmentation mask
            - pcd_pts: list of np.ndarrays for each segmented point cloud
            - pcd_colors: list of np.ndarrays for the colors corresponding
                to the points of each segmented pointcloud
            - pcd_full: fused point cloud with points from each camera view
        """
        rgbs = []
        depths = []
        segs = []
        obj_pcd_pts = []
        obj_pcd_colors = []
        table_pcd_pts = []
        table_pcd_colors = []

        for cam in self.cams:
            rgb, depth = cam.get_images(
                get_rgb=True,
                get_depth=True
            )

            # use all instances segmented from mask rcnn
            det_seg = self.detectron_seg(rgb[:, :, ::-1])
            det_seg_full = np.zeros((det_seg.shape[1], det_seg.shape[2]))
            for k in range(det_seg.shape[0]):
                det_seg_full = np.logical_or(det_seg[k, :, :], det_seg_full)

            # visualize the masks
            # for ind in range(det_seg.shape[0]):
            #     det_res = cv2.bitwise_and(rgb, rgb, mask=det_seg[ind, :, :].astype(np.uint8))
            #     plt.imshow(det_res)
            #     plt.show()   
            
            det_res = cv2.bitwise_and(rgb, rgb, mask=det_seg_full.astype(np.uint8))
            plt.imshow(det_res)
            plt.show()            

            pts_raw, colors_raw = cam.get_pcd(
                in_world=True,
                filter_depth=False,
                depth_max=depth_max
            )

            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

            lower_red = np.array([0, 75, 50])
            upper_red = np.array([60, 255, 180])            

            seg = cv2.inRange(hsv, lower_red, upper_red)
            if color_seg:
                flat_seg = seg.flatten()
            else:          
                flat_seg = det_seg_full.flatten()

            # # Vizualize the mask
            # plt.imshow(mask, cmap='gray')
            # plt.show()

            # res = cv2.bitwise_and(rgb, rgb, mask=seg)
            # plt.imshow(res)
            # plt.show()

            obj_pts, obj_colors = pts_raw[np.where(flat_seg)[0], :], colors_raw[np.where(flat_seg)[0], :]            
            obj_pts, obj_colors = self.real_seg(obj_pts, obj_colors)

            rgbs.append(copy.deepcopy(rgb))
            depths.append(copy.deepcopy(depth))
            segs.append(copy.deepcopy(seg))

            obj_pcd_pts.append(obj_pts)
            obj_pcd_colors.append(obj_colors)

            # segment the table by using the rest of the points that are not the object, and then cropping
            not_obj_seg = np.where(np.logical_not(flat_seg))[0]
            not_obj_pts, not_obj_colors = pts_raw[not_obj_seg, :], colors_raw[not_obj_seg, :]
            table_pts, table_colors = self.real_seg(not_obj_pts, not_obj_colors, table=True)

            # table_pcd = open3d.geometry.PointCloud()
            # table_pcd.points = open3d.utility.Vector3dVector(table_pts)
            # table_pcd.colors = open3d.utility.Vector3dVector(table_colors) 
            # open3d.visualization.draw_geometries([table_pcd])

            table_pcd_pts.append(table_pts)
            table_pcd_colors.append(table_colors)
            # if robot_table_id is not None:
            #     if robot_table_id[1] == -1:
            #         table_inds = np.where(flat_seg == robot_table_id[0])
            #     else:
            #         robot_id, table_id = robot_table_id[0], robot_table_id[1]
            #         table_val = robot_id + (table_id+1) << 24
            #         table_inds = np.where(flat_seg == table_val)
            #     table_pts, table_colors = pts_raw[table_inds[0], :], colors_raw[table_inds[0], :]
            #     keep_inds = np.where(table_pts[:, 0] > 0.0)[0]
            #     table_pts = table_pts[keep_inds, :]
            #     table_colors = table_colors[keep_inds, :]
            #     table_pcd_pts.append(table_pts)
            #     table_pcd_colors.append(table_colors)         

        pcd = open3d.geometry.PointCloud()

        obj_pcd_pts_cat = np.concatenate(obj_pcd_pts, axis=0)
        obj_pcd_colors_cat = np.concatenate(obj_pcd_colors, axis=0)

        pcd.points = open3d.utility.Vector3dVector(obj_pcd_pts_cat)
        pcd.colors = open3d.utility.Vector3dVector(obj_pcd_colors_cat / 255.0)
        pcd.estimate_normals()

        total_pts = obj_pcd_pts_cat.shape[0]
        down_pcd = pcd.uniform_down_sample(int(total_pts/downsampled_pcd_size))

        obs_dict = {}
        obs_dict['rgb'] = rgbs
        obs_dict['depth'] = depths
        obs_dict['seg'] = segs
        obs_dict['pcd_pts'] = obj_pcd_pts
        obs_dict['pcd_colors'] = obj_pcd_colors
        obs_dict['pcd_normals'] = np.asarray(pcd.normals)
        obs_dict['table_pcd_pts'] = table_pcd_pts
        obs_dict['table_pcd_colors'] = table_pcd_colors
        obs_dict['down_pcd_pts'] = np.asarray(down_pcd.points)
        obs_dict['down_pcd_normals'] = np.asarray(down_pcd.normals)
        obs_dict['center_of_mass'] = pcd.get_center()
        return obs_dict, pcd

    def real_seg(self, pts_raw, colors_raw, table=False):
        x_min, x_max = 0.1, 0.6
        y_min, y_max = -0.45, 0.45
        if table:
            z_min, z_max = -0.01, 0.01
        else:
            z_min, z_max = 0.01, 0.5

        pts, colors = pts_raw, colors_raw
        obj_inds = [np.arange(pts_raw.shape[0], dtype=np.int64)]

        x_inds = np.where(np.logical_and(pts[:, 0] > x_min, pts[:, 0] < x_max))[0]
        pts, colors = pts[x_inds, :], colors[x_inds, :]
        y_inds = np.where(np.logical_and(pts[:, 1] > y_min, pts[:, 1] < y_max))[0]
        pts, colors = pts[y_inds, :], colors[y_inds, :]
        z_inds = np.where(np.logical_and(pts[:, 2] > z_min, pts[:, 2] < z_max))[0]
        pts, colors = pts[z_inds, :], colors[z_inds, :]

        return pts, colors

    def _setup_detectron(self):
        self._detectron_obs_dir = '/tmp/detectron/observations'
        self._detectron_pred_dir = '/tmp/detectron/predictions'
        if not osp.exists(self._detectron_obs_dir):
            os.makedirs(self._detectron_obs_dir)
        if not osp.exists(self._detectron_pred_dir):
            os.makedirs(self._detectron_pred_dir)

        obs_fnames, pred_fnames = os.listdir(self._detectron_obs_dir), os.listdir(self._detectron_pred_dir)
        if len(obs_fnames):
            for fname in obs_fnames:
                os.remove(osp.join(self._detectron_obs_dir, fname))
        if len(pred_fnames):
            for fname in pred_fnames:
                os.remove(osp.join(self._detectron_pred_dir, fname))

        self.samples_count = 0
        # TODO send ping to the directories to make sure things are working

    def detectron_seg(self, rgb):
        """
        Function to use filesystem as a pub/sub to send out the observed RGB image
        and return a segmentation mask of the object in the image

        Args:
            rgb (np.ndarray): W x H x 3 np.ndarray RGB image

        Returns:
            np.ndarray: W x H x 1 segmentation mask of the object in the image
        """    
        self.samples_count += 1
        obs = copy.deepcopy(rgb)
        obs_fname = osp.join(self._detectron_obs_dir, str(self.samples_count) + '.jpg')
        im = Image.fromarray(obs)
        im.save(obs_fname)

        pred_fname = osp.join(self._detectron_pred_dir, str(self.samples_count) + '.npz')
        got_file = False
        start = time.time()
        while True:
            pred_fnames = os.listdir(self._detectron_pred_dir)
            if len(pred_fnames):
                try:
                    pred = np.load(pred_fname)
                    got_file = True
                except:
                    pass
            if got_file or (time.time() - start > 300):
                break
            time.sleep(0.05)
        os.remove(pred_fname)

        # TODO process segmentation mask from prediction
        seg = copy.deepcopy(pred['pred']).astype(np.uint16)
        return seg

class DataManager(object):
    def __init__(self, save_path):
        self.save_path = save_path
        self.pickle_dir = 'pkl'
        self.img_dir = 'img'
        self.pcd_dir = 'pcd'
        self.depth_scale = 1000.0

    def make_paths(self, raw_fname):

        pcd_fname = os.path.join(
            self.save_path,
            raw_fname,
            self.pcd_dir, raw_fname + '_pcd.pkl')

        depth_fnames = [raw_fname + '_depth_%d.png' % j for j in range(3)]
        rgb_fnames = [raw_fname + '_rgb_%d.png' % j for j in range(3)]
        seg_fnames = [raw_fname + '_seg_%d.pkl' % j for j in range(3)]

        if not os.path.exists(os.path.join(self.save_path, raw_fname, self.pickle_dir)):
            os.makedirs(os.path.join(self.save_path, raw_fname, self.pickle_dir))
        if not os.path.exists(os.path.join(self.save_path, raw_fname, self.img_dir)):
            os.makedirs(os.path.join(self.save_path, raw_fname, self.img_dir))
        if not os.path.exists(os.path.join(self.save_path, raw_fname, self.pcd_dir)):
            os.makedirs(os.path.join(self.save_path, raw_fname, self.pcd_dir))

        return pcd_fname, depth_fnames, rgb_fnames, seg_fnames

    def save_observation(self, data_dict, filename):
        raw_fname = filename

        pcd_fname, depth_fnames, rgb_fnames, seg_fnames = self.make_paths(raw_fname)

        pkl_data = {}
        for key in data_dict.keys():
            if not key == 'obs':
                pkl_data[key] = copy.deepcopy(data_dict[key])

        with open(os.path.join(self.save_path, raw_fname, self.pickle_dir, raw_fname+'.pkl'), 'wb') as pkl_f:
            pickle.dump(pkl_data, pkl_f)

        if 'obs' in data_dict.keys():
            # save depth
            for k, fname in enumerate(rgb_fnames):
                rgb_fname = os.path.join(self.save_path, raw_fname, self.img_dir, rgb_fnames[k])
                depth_fname = os.path.join(self.save_path, raw_fname, self.img_dir, depth_fnames[k])
                seg_fname = os.path.join(self.save_path, raw_fname, self.img_dir, seg_fnames[k])

                # save depth
                sdepth = data_dict['obs']['depth'][k] * self.depth_scale
                cv2.imwrite(depth_fname, sdepth.astype(np.uint16))

                # save rgb
                cv2.imwrite(rgb_fname, data_dict['obs']['rgb'][k])

                # save seg
                with open(seg_fname, 'wb') as f:
                    pickle.dump(data_dict['obs']['seg'][k], f, protocol=pickle.HIGHEST_PROTOCOL)

            # save pointcloud arrays as .pkl with high protocol
            pcd_pts = data_dict['obs']['pcd_pts']
            pcd_colors = data_dict['obs']['pcd_colors']
            down_pcd_pts = data_dict['obs']['down_pcd_pts']
            table_pcd_pts = data_dict['obs']['table_pcd_pts']
            table_pcd_colors = data_dict['obs']['table_pcd_colors']

            pcd = {}
            pcd['pts'] = pcd_pts
            pcd['colors'] = pcd_colors
            pcd['down_pts'] = down_pcd_pts
            pcd['table_pts'] = table_pcd_pts
            pcd['table_colors'] = table_pcd_colors
            with open(pcd_fname, 'wb') as f:
                pickle.dump(pcd, f, protocol=pickle.HIGHEST_PROTOCOL)


class MultiBlockManager(object):
    def __init__(self, cuboid_path, cuboid_sampler,
                 robot_id, table_id, r_gel_id, l_gel_id, fname_prefix='test_cuboid_smaller_'):
        self.sampler = cuboid_sampler
        self.cuboid_path = cuboid_path

        self.r_gel_id = r_gel_id
        self.l_gel_id = l_gel_id
        self.robot_id = robot_id
        self.table_id = table_id

        self.gel_ids = [self.r_gel_id, self.l_gel_id]

        self.cuboid_fname_prefix = fname_prefix
        self.setup_block_set()

    def setup_block_set(self):
        self.cuboid_fnames = []
        for fname in os.listdir(self.cuboid_path):
            if fname.startswith(self.cuboid_fname_prefix):
                self.cuboid_fnames.append(os.path.join(self.cuboid_path,
                                                       fname))
        print('Loaded cuboid files: ')
        # print(self.cuboid_fnames)

    def get_cuboid_fname(self):
        ind = np.random.randint(len(self.cuboid_fnames))
        return self.cuboid_fnames[ind]

    def filter_collisions(self, obj_id, goal_obj_id=None):
        for gel_id in self.gel_ids:
            if goal_obj_id is not None:
                p.setCollisionFilterPair(self.robot_id,
                                         goal_obj_id,
                                         gel_id,
                                         -1,
                                         enableCollision=False)
            p.setCollisionFilterPair(self.robot_id,
                                     obj_id,
                                     gel_id,
                                     -1,
                                     enableCollision=True)
        p.setCollisionFilterPair(self.robot_id,
                                 obj_id,
                                 self.table_id,
                                 -1,
                                 enableCollision=True)
        if goal_obj_id is not None:
            for jnt_id in range(self.table_id):
                p.setCollisionFilterPair(self.robot_id, goal_obj_id, jnt_id, -1, enableCollision=False)
            p.setCollisionFilterPair(self.robot_id,
                                     goal_obj_id,
                                     self.table_id,
                                     -1,
                                     enableCollision=True)

    def robot_collisions_filter(self, obj_id, enable=True):
        for jnt_id in range(self.table_id):
            p.setCollisionFilterPair(self.robot_id, obj_id, jnt_id, -1, enableCollision=enable)
        p.setCollisionFilterPair(self.robot_id,
                                    obj_id,
                                    self.table_id,
                                    -1,
                                    enableCollision=True)


class GoalVisual():
    def __init__(self, trans_box_lock, object_id, pb_client,
                 goal_init, show_init=True):
        self.trans_box_lock = trans_box_lock
        self.pb_client = pb_client
        self.update_goal_obj(object_id)

        if show_init:
            self.update_goal_state(goal_init)

    def visualize_goal_state(self):
        while True:
            self.trans_box_lock.acquire()
            p.resetBasePositionAndOrientation(
                self.object_id,
                [self.goal_pose[0], self.goal_pose[1], self.goal_pose[2]],
                self.goal_pose[3:],
                physicsClientId=self.pb_client)
            self.trans_box_lock.release()
            time.sleep(0.01)

    def update_goal_state(self, goal):
        self.trans_box_lock.acquire()
        self.goal_pose = goal
        p.resetBasePositionAndOrientation(
            self.object_id,
            [self.goal_pose[0], self.goal_pose[1], self.goal_pose[2]],
            self.goal_pose[3:],
            physicsClientId=self.pb_client)
        self.trans_box_lock.release()

    def update_goal_obj(self, obj_id):
        self.object_id = obj_id
        self.color_data = p.getVisualShapeData(obj_id)[0][7]

    def hide_goal_obj(self):
        color = [self.color_data[0],
                 self.color_data[1],
                 self.color_data[2],
                 0]
        p.changeVisualShape(self.object_id, -1, rgbaColor=color)

    def show_goal_obj(self):
        p.changeVisualShape(self.object_id, -1, rgbaColor=self.color_data)


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


def main(args):
    cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    rospy.init_node('MacroActions')
    signal.signal(signal.SIGINT, signal_handler)

    data = {}
    data['saved_data'] = []
    data['metadata'] = {}

    data_seed = args.np_seed
    np.random.seed(data_seed)

    yumi_ar = Robot('yumi_palms',
                    pb=True,
                    pb_cfg={'gui': args.visualize},
                    arm_cfg={'self_collision': False,
                             'seed': data_seed})

    r_gel_id = cfg.RIGHT_GEL_ID
    l_gel_id = cfg.LEFT_GEL_ID

    alpha = cfg.ALPHA
    K = cfg.GEL_CONTACT_STIFFNESS
    restitution = cfg.GEL_RESTITUION

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        r_gel_id,
        restitution=restitution,
        contactStiffness=K,
        contactDamping=alpha*K,
        rollingFriction=args.rolling
    )

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        l_gel_id,
        restitution=restitution,
        contactStiffness=K,
        contactDamping=alpha*K,
        rollingFriction=args.rolling
    )
    dynamics_info = {}
    dynamics_info['contactDamping'] = alpha*K
    dynamics_info['contactStiffness'] = K
    dynamics_info['rollingFriction'] = args.rolling
    dynamics_info['restitution'] = restitution

    yumi_gs = YumiCamsGS(
        yumi_ar,
        cfg,
        exec_thread=False,
        sim_step_repeat=args.step_repeat)

    for _ in range(10):
        yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    obj_id = yumi_ar.pb_client.load_urdf(
        args.config_package_path +
        'descriptions/urdf/'+args.object_name+'.urdf',
        cfg.OBJECT_POSE_3[0:3],
        cfg.OBJECT_POSE_3[3:]
    )

    goal_obj_id = yumi_ar.pb_client.load_urdf(
        args.config_package_path +
        'descriptions/urdf/'+args.object_name+'_trans.urdf',
        cfg.OBJECT_POSE_3[0:3],
        cfg.OBJECT_POSE_3[3:]
    )
    p.setCollisionFilterPair(yumi_ar.arm.robot_id, goal_obj_id, r_gel_id, -1, enableCollision=False)
    p.setCollisionFilterPair(obj_id, goal_obj_id, -1, -1, enableCollision=False)
    p.setCollisionFilterPair(yumi_ar.arm.robot_id, obj_id, r_gel_id, -1, enableCollision=True)
    p.setCollisionFilterPair(yumi_ar.arm.robot_id, obj_id, 27, -1, enableCollision=True)

    yumi_ar.pb_client.reset_body(
        obj_id,
        cfg.OBJECT_POSE_3[:3],
        cfg.OBJECT_POSE_3[3:])

    yumi_ar.pb_client.reset_body(
        goal_obj_id,
        cfg.OBJECT_POSE_3[:3],
        cfg.OBJECT_POSE_3[3:])

    primitive_name = args.primitive

    mesh_file = args.config_package_path + 'descriptions/meshes/objects/' + args.object_name + '_experiments.stl'
    exp_single = SingleArmPrimitives(
        cfg,
        yumi_ar.pb_client.get_client_id(),
        obj_id,
        mesh_file)
    exp_double = DualArmPrimitives(
        cfg,
        yumi_ar.pb_client.get_client_id(),
        obj_id,
        mesh_file)
    if primitive_name == 'grasp':
        exp_running = exp_double
    else:
        exp_running = exp_single

    # setup macro_planner
    action_planner = ClosedLoopMacroActions(
        cfg,
        yumi_gs,
        obj_id,
        yumi_ar.pb_client.get_client_id(),
        args.config_package_path,
        replan=args.replan,
        object_mesh_file=mesh_file
    )

    data['metadata']['mesh_file'] = mesh_file
    data['metadata']['cfg'] = cfg
    data['metadata']['dynamics'] = dynamics_info
    data['metadata']['cam_cfg'] = yumi_gs.cam_setup_cfg
    data['metadata']['step_repeat'] = args.step_repeat

    delta_z_height = 0.95
    with open(args.config_package_path+'descriptions/urdf/'+args.object_name+'.urdf', 'rb') as f:
        urdf_txt = f.read()

    data['metadata']['object_urdf'] = urdf_txt
    data['metadata']['delta_z_height'] = delta_z_height
    data['metadata']['step_repeat'] = args.step_repeat
    data['metadata']['seed'] = data_seed

    metadata = data['metadata']

    if args.multi:
        cuboid_sampler = CuboidSampler(
            '/root/catkin_ws/src/primitives/objects/cuboids/nominal_cuboid.stl',
            pb_client=yumi_ar.pb_client)
        cuboid_fname_template = '/root/catkin_ws/src/primitives/objects/cuboids/'

        cuboid_manager = MultiBlockManager(
            cuboid_fname_template,
            cuboid_sampler,
            robot_id=yumi_ar.arm.robot_id,
            table_id=27,
            r_gel_id=r_gel_id,
            l_gel_id=l_gel_id)

        yumi_ar.pb_client.remove_body(obj_id)
        yumi_ar.pb_client.remove_body(goal_obj_id)

        cuboid_fname = cuboid_manager.get_cuboid_fname()

        obj_id, sphere_ids, mesh, goal_obj_id = \
            cuboid_sampler.sample_cuboid_pybullet(
                cuboid_fname,
                goal=True,
                keypoints=False)

        cuboid_manager.filter_collisions(obj_id, goal_obj_id)
        action_planner.update_object(obj_id, mesh_file)

    trans_box_lock = threading.RLock()
    goal_viz = GoalVisual(
        trans_box_lock,
        goal_obj_id,
        action_planner.pb_client,
        cfg.OBJECT_POSE_3)

    pickle_path = os.path.join(
        args.data_dir,
        primitive_name,
        args.experiment_name
    )

    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)

    data_manager = DataManager(pickle_path)

    if args.save_data:
        with open(os.path.join(pickle_path, 'metadata.pkl'), 'wb') as mdata_f:
            pickle.dump(metadata, mdata_f)

    obs, pcd = yumi_gs.get_observation(obj_id=obj_id)

    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_trials',
        type=int
    )

    parser.add_argument(
        '--step_repeat',
        type=int,
        default=10
    )

    parser.add_argument(
        '--save_data',
        action='store_true'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/'
    )

    parser.add_argument(
        '--experiment_name',
        type=str,
        default='sample_experiment'
    )

    parser.add_argument(
        '--config_package_path',
        type=str,
        default='/root/catkin_ws/src/config/')

    parser.add_argument(
        '--example_config_path',
        type=str,
        default='config')

    parser.add_argument(
        '--primitive',
        type=str,
        default='pull',
        help='which primitive to plan')

    parser.add_argument(
        '--simulate',
        type=int,
        default=1)

    parser.add_argument(
        '-o',
        '--object',
        action='store_true')

    parser.add_argument(
        '-re',
        '--replan',
        action='store_true',
        default=False)

    parser.add_argument(
        '--object_name',
        type=str,
        default='realsense_box')

    parser.add_argument(
        '-ex',
        '--execute_thread',
        action='store_true'
    )

    parser.add_argument(
        '--debug', action='store_true'
    )

    parser.add_argument(
        '-r', '--rolling',
        type=float, default=0.0,
        help='rolling friction value for pybullet sim'
    )

    parser.add_argument(
        '-v', '--visualize',
        action='store_true'
    )

    parser.add_argument(
        '--np_seed', type=int,
        default=0
    )

    parser.add_argument(
        '--multi', action='store_true'
    )

    parser.add_argument(
        '--num_obj_samples', type=int, default=10
    )

    args = parser.parse_args()
    main(args)
