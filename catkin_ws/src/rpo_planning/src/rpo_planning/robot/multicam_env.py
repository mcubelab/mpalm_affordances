import os, os.path as osp
import time
import copy
import open3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from yacs.config import CfgNode as CN 
from PIL import Image
import scipy.misc

from airobot.utils.ros_util import read_cam_ext
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from airobot.sensor.camera.rgbdcam_real import RGBDCameraReal

from rpo_planning.robot.pybullet_ros import YumiPybullet 
from rpo_planning.robot.real_ros import YumiReal


class YumiMulticamPybullet(YumiPybullet):
    """
    Child class of YumiGelslimPybullet with additional functions
    for setting up multiple cameras in the pybullet scene
    and getting observations of various types
    """
    def __init__(self, yumi_ar, cfg, n_cams=4, exec_thread=True, sim_step_repeat=10):
        """
        Constructor, sets up base class and additional camera setup
        configuration parameters.

        Args:
            yumi_ar (airobot Robot): Instance of PyBullet simulated robot, from
                airobot library
            cfg (YACS CfgNode): Configuration parameters
            exec_thread (bool, optional): Whether or not to start the
                background joint position control thread. Defaults to True.
            sim_step_repeat (int, optional): Number of simulation steps
                to take each time the desired joint position value is
                updated. Defaults to 10
        """
        super(YumiMulticamPybullet, self).__init__(yumi_ar,
                                         cfg,
                                         exec_thread=exec_thread,
                                         sim_step_repeat=sim_step_repeat)

        self.cams = []
        for _ in range(4):
            self.cams.append(RGBDCameraPybullet(cfgs=self._camera_cfgs(),
                                                pb_client=yumi_ar.pb_client))

        self.n_cams = n_cams
        self.cam_setup_cfg = {}
        # self.cam_setup_cfg['focus_pt'] = [self.cfg.OBJECT_POSE_3[:3]]*3
        # self.cam_setup_cfg['dist'] = [0.7, 0.7, 0.75]
        # self.cam_setup_cfg['yaw'] = [30, 150, 270]
        # self.cam_setup_cfg['pitch'] = [-45, -45, -70]
        # self.cam_setup_cfg['roll'] = [0, 0, 0]
        self.cam_setup_cfg['focus_pt'] = [self.cfg.CAMERA_FOCUS]*self.n_cams
        self.cam_setup_cfg['dist'] = [0.8, 0.8, 0.8, 0.8]  # TODO make based on n_cams
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
        Function to set up pybullet cameras in the simulated environment
        """
        for i in range(self.n_cams):
            self.cams[i].setup_camera(
                focus_pt=self.cam_setup_cfg['focus_pt'][i],
                dist=self.cam_setup_cfg['dist'][i],
                yaw=self.cam_setup_cfg['yaw'][i],
                pitch=self.cam_setup_cfg['pitch'][i],
                roll=self.cam_setup_cfg['roll'][i]
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


class YumiMulticamReal(YumiReal):
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
        super(YumiMulticamReal, self).__init__(yumi_ar, cfg)
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