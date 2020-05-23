import os, sys
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
from IPython import embed

from airobot import Robot
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions
from yumi_pybullet_ros import YumiGelslimPybullet

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

    def get_observation(self, obj_id, depth_max=1.0, downsampled_pcd_size=100, robot_table_id=None):
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
            rgb, depth, seg = cam.get_images(
                get_rgb=True,
                get_depth=True,
                get_seg=True
            )

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
                 robot_id, table_id, r_gel_id, l_gel_id):
        self.sampler = cuboid_sampler
        self.cuboid_path = cuboid_path

        self.r_gel_id = r_gel_id
        self.l_gel_id = l_gel_id
        self.robot_id = robot_id
        self.table_id = table_id

        self.gel_ids = [self.r_gel_id, self.l_gel_id]

        self.cuboid_fname_prefix = 'test_cuboid_smaller_'
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


class GoalVisual():
    def __init__(self, trans_box_lock, object_id, pb_client,
                 goal_init, show_init=True):
        self.trans_box_lock = trans_box_lock
        self.pb_client = pb_client
        self.object_id = object_id

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
