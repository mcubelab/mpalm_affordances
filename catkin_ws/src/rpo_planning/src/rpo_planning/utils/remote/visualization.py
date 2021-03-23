import time
import pybullet as p
import numpy as np
import copy
import open3d
import cv2
# from multiprocessing import Process, Queue, Manager
import threading
from queue import Queue
from yacs.config import CfgNode as CN

from airobot import set_log_level, log_debug, log_info, log_warn, log_critical
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet


class MultiCams:
    """
    Class for easily obtaining simulated camera image observations in pybullet
    """
    def __init__(self, pb_client, focus_pt=[0.0, 0.0, 0.0], n_cams=4,
                 cam_setup_cfg=None):
        """
        Constructor, sets up base class and additional camera setup
        configuration parameters.

        Args:
            robot (airobot Robot): Instance of PyBullet simulated robot, from
                airobot library
            n_cams (int): Number of cameras to put in the world
        """
        self.n_cams = n_cams
        self.pb_client = pb_client
        self.cams = []
        for _ in range(n_cams):
            self.cams.append(RGBDCameraPybullet(cfgs=self._camera_cfgs(),
                                                pb_client=self.pb_client))

        default_distance = 1.5 
        default_yaw_angles = [30, 150, 210, 330]
        if cam_setup_cfg is None:
            self.cam_setup_cfg = {}

            self.cam_setup_cfg['focus_pt'] = [focus_pt] * n_cams
            self.cam_setup_cfg['dist'] = [default_distance] * n_cams
            # self.cam_setup_cfg['yaw'] = [30, 150, 210, 330]
            self.cam_setup_cfg['yaw'] = default_yaw_angles[:n_cams] 
            self.cam_setup_cfg['pitch'] = [-35] * n_cams
            self.cam_setup_cfg['roll'] = [0] * n_cams
        else:
            self.cam_setup_cfg = cam_setup_cfg

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
        for i, cam in enumerate(self.cams):
            cam.setup_camera(
                focus_pt=self.cam_setup_cfg['focus_pt'][i],
                dist=self.cam_setup_cfg['dist'][i],
                yaw=self.cam_setup_cfg['yaw'][i],
                pitch=self.cam_setup_cfg['pitch'][i],
                roll=self.cam_setup_cfg['roll'][i]                
            )

    def get_observation(self):
        """
        Function to get an observation from the pybullet scene. Gets
        an RGB-D images 

        Returns:
            - dict: Contains observation data, with keys for
                - rgb: list of np.ndarrays for each RGB image
                - depth: list of np.ndarrays for each depth image
                - seg: list of np.ndarrays for each segmentation mask
        """
        rgbs = []
        depths = []
        segs = []

        for i, cam in enumerate(self.cams):
            rgb, depth, seg = cam.get_images(
                get_rgb=True,
                get_depth=True,
                get_seg=True,
            )
            rgbs.append(rgb)
            depths.append(depth)
            segs.append(seg)

        obs = {}
        obs['rgb'] = rgbs
        obs['depth'] = depths
        obs['seg'] = segs
        return obs 


class PyBulletVideoRecorder:
    """
    Class for obtaining a video recording of a PyBullet environment using
    PyBullet's simulated camera renderer to obtain a sequence of images
    and opencv to write the image sequence to a video
    """
    def __init__(self, pb_client):
        self.pb_client = pb_client
        self._recording = False 

    def recorder(self, pb_client, loop_t):
        """
        Main function used in pybullet recording thread

        Args:
            pb_client (airobot.utils.pb_util.BulletClient): Interface to
                pybullet physics simulation through airobot library.
            loop_t (float): Amount of time to pass each time a frame is captured
        """
        cams = MultiCams(
            pb_client, 
            focus_pt=[0.25, 0.0, 0.1],
            n_cams=1) 

        # initialize buffers for images to come in
        rgb_frames = []
        depth_frames = []
        seg_frames = []

        loop_time = time.time()
        while True:
            if self.stop_thread:
                break

            if time.time() - loop_time > loop_t:
                loop_time = time.time()
                obs = cams.get_observation()
                rgb_list = obs['rgb']
                depth_list = obs['depth']
                seg_list = obs['seg']

                rgb_frames.append(rgb_list[0])
                depth_frames.append(depth_list[0])
                seg_frames.append(seg_list[0])
        self.frames = rgb_frames
        return

    def record(self, loop_t=1):
        """
        Function to begin recording in a background thread

        Args:
            video_name (str): Name of video file to save
        """
        self.p = threading.Thread(
            target=self.recorder,
            args=(
                self.pb_client,
                loop_t,
            )
        ) 
        self.p.daemon = True
        self.stop_thread = False
        self._recording = True
        self.p.start()
    
    def stop(self, save=True, video_name=None, fps=10, return_frames=False):
        """
        Function to stop recording and save the video that has
        been obtained

        Args:
            save (bool): If True, save data that has been written
                to the image buffers. Otherwise just make sure
                recording has stopped.
            return_frames (bool): If True, function should return the 
                sequence of image frames as a list of np.ndarray's

        Returns:
            list (or None): Sequence of image frames
        """
        if self._recording:
            self.stop_thread = True
            self.p.join()
            self._recording = False
            if save and video_name is not None:
                return self.save(video_name=video_name, fps=fps, return_frames=return_frames)
            else:
                return None
        else:
            self._recording = False
            log_warn('Video recorder: Recording was never started, not saving and returning None')
            return None
    
    def save(self, video_name, fps=10, return_frames=False):
        """
        Function to save whatever files are currently in the buffer

        Args:
            video_name (str): Name of file to save
            fps (int): Frames per second in the video writer
            return_frames (bool): If True, function should return the 
                sequence of image frames as a list of np.ndarray's

        Returns:
            list (or None): Sequence of image frames
        """
        self.stop()
        if len(self.frames) > 0:
            rgb_frames = self.frames
            if len(rgb_frames) < 1:
                log_warn('Video recorder: Did not capture any frames, not saving and returning None')
                return None

            size = rgb_frames[0].shape[:2]
            rgb_video = cv2.VideoWriter(
                video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))
            for i, frame in enumerate(rgb_frames):
                rgb_video.write(frame[:, :, ::-1])  # convert from rgb to bgr
            rgb_video.release()
            if return_frames:
                return rgb_frames    
        else:
            log_warn('Video recorder: Did not capture any frames, not saving and returning None')
            return None

if __name__ == '__main__':
    def sin_wave(t, f, A):
        """
        Return the sine-wave value at time step t.

        Args:
            t (float): time
            f (float): frequency
            A (float): amplitude

        Returns:
            a sin-wave value

        """
        return A * np.cos(2 * np.pi * f * t)


    def main(robot):
        """
        This function demonstrates how to move the robot arm
        to the desired joint positions.
        """
        robot.arm.go_home(ignore_physics=True)

        A = 0.4
        f = 0.4
        start_time = time.time()
        while True:
            elapsed_time = time.time() - start_time
            vels = [sin_wave(elapsed_time, f, A)] * robot.arm.arm_dof
            robot.arm.set_jvel(vels)
            time.sleep(0.01)
            if time.time() - start_time > 10:
                break
        return

    from airobot import Robot
    robot = Robot('ur5e', pb_cfg={'gui': False, 'opengl_render': False})

    recorder = PyBulletVideoRecorder(
        pb_client=robot.pb_client
    )

    for i in range(5):
        name = 'video_%d.avi' % i
        recorder.record(loop_t=1)
        main(robot)
        recorder.stop()
        output = recorder.save(name, return_frames=True)
    