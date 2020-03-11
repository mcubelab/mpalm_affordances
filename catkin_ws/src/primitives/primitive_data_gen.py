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
from airobot.utils import pb_util
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions, YumiGelslimPybulet
from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

from yacs.config import CfgNode as CN
from closed_loop_experiments_cfg import get_cfg_defaults
from proc_gen_cuboids import CuboidSampler


class YumiCamsGS(YumiGelslimPybulet):
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
        for i in range(3):
            self.cams.append(RGBDCameraPybullet(cfgs=self._camera_cfgs()))

        self.cam_setup_cfg = {}
        self.cam_setup_cfg['focus_pt'] = [self.cfg.OBJECT_POSE_3[:3]]*3
        self.cam_setup_cfg['dist'] = [0.7, 0.7, 0.75]
        self.cam_setup_cfg['yaw'] = [30, 150, 270]
        self.cam_setup_cfg['pitch'] = [-45, -45, -70]
        self.cam_setup_cfg['roll'] = [0, 0, 0]

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

    def get_observation(self, obj_id, depth_max=1.0):
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

        for cam in self.cams:
            rgb, depth = cam.get_images(
                get_rgb=True, get_depth=True
            )

            seg = cam.get_segmentation_mask()

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

        pcd = open3d.geometry.PointCloud()

        obj_pcd_pts_cat = np.concatenate(obj_pcd_pts, axis=0)
        obj_pcd_colors_cat = np.concatenate(obj_pcd_colors, axis=0)

        pcd.points = open3d.utility.Vector3dVector(obj_pcd_pts_cat)
        pcd.colors = open3d.utility.Vector3dVector(obj_pcd_colors_cat / 255.0)

        obs_dict = {}
        obs_dict['rgb'] = rgbs
        obs_dict['depth'] = depths
        obs_dict['seg'] = segs
        obs_dict['pcd_pts'] = obj_pcd_pts
        obs_dict['pcd_colors'] = obj_pcd_colors
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
                    # cv2.imwrite(seg_fname, data['obs']['seg'][i])
                    with open(seg_fname, 'wb') as f:
                        pickle.dump(data_dict['obs']['seg'][k], f, protocol=pickle.HIGHEST_PROTOCOL)

                # save pointcloud as pcd
                # pcd = open3d.geometry.PointCloud()
                # pcd.points = open3d.utility.Vector3dVector(np.concatenate(data['obs']['pcd_pts'], axis=0))
                # pcd.colors = open3d.utility.Vector3dVector(np.concatenate(data['obs']['pcd_colors'], axis=0) / 255.0)
                # open3d.io.write_point_cloud(pcd_fname, pcd)

                # save pointcloud arrays as .pkl with high protocol
                pcd_pts = data_dict['obs']['pcd_pts']
                pcd_colors = data_dict['obs']['pcd_colors']

                pcd = {}
                pcd['pts'] = pcd_pts
                pcd['colors'] = pcd_colors
                with open(pcd_fname, 'wb') as f:
                    pickle.dump(pcd, f, protocol=pickle.HIGHEST_PROTOCOL)


class GoalVisual():
    def __init__(self, trans_box_lock, object_id, pb_client, goal_init):
        self.trans_box_lock = trans_box_lock
        self.pb_client = pb_client
        self.object_id = object_id

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

def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


def worker_yumi(child_conn, work_queue, result_queue, cfg, args):
    while True:
        # print("here!")
        try:
            if not child_conn.poll(0.0001):
                continue
            msg = child_conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if msg == "RESET":
            # yumi = Robot('yumi', pb=True, arm_cfg={'render': True, 'self_collision': False})
            # client_id = p.connect(p.DIRECT)
            # print("\n\nfinished worker construction\n\n")
            yumi_ar = Robot('yumi',
                            pb=True,
                            arm_cfg={'render': True, 'self_collision': False})

            yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT)

            gel_id = 12

            alpha = 0.01
            K = 500

            p.changeDynamics(
                yumi_ar.arm.robot_id,
                gel_id,
                restitution=0.99,
                contactStiffness=K,
                contactDamping=alpha*K,
                rollingFriction=args.rolling
            )

            # setup yumi_gs
            yumi_gs = YumiGelslimPybulet(yumi_ar, cfg, exec_thread=args.execute_thread, sim_step_repeat=args.step_repeat)


            box_id = pb_util.load_urdf(
                args.config_package_path +
                'descriptions/urdf/'+args.object_name+'.urdf',
                cfg.OBJECT_POSE_3[0:3],
                cfg.OBJECT_POSE_3[3:]
            )
            trans_box_id = pb_util.load_urdf(
                args.config_package_path +
                'descriptions/urdf/'+args.object_name+'_trans.urdf',
                cfg.OBJECT_POSE_3[0:3],
                cfg.OBJECT_POSE_3[3:]
            )

            # setup macro_planner
            action_planner = ClosedLoopMacroActions(
                cfg,
                yumi_gs,
                box_id,
                pb_util.PB_CLIENT,
                args.config_package_path,
                replan=args.replan
            )
            continue
        if msg == "HOME":
            yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)
            continue
        if msg == "OBJECT_POSE":
            obj_pos_world = list(p.getBasePositionAndOrientation(
                box_id,
                pb_util.PB_CLIENT)[0])
            obj_ori_world = list(p.getBasePositionAndOrientation(
                box_id,
                pb_util.PB_CLIENT)[1])

            obj_pose_world = util.list2pose_stamped(
                obj_pos_world + obj_ori_world)
            work_queue.put(obj_pose_world)
            continue
        if msg == "SAMPLE":
            # try:
            #     example_args = work_queue.get(block=True)
            #     primitive_name = example_args['primitive_name']
            #     result = action_planner.execute(primitive_name, example_args)
            #     work_queue.put(result)
            # except work_queue.Empty:
            #     continue
            manipulated_object = None
            object_pose1_world = util.list2pose_stamped(cfg.OBJECT_INIT)
            object_pose2_world = util.list2pose_stamped(cfg.OBJECT_FINAL)
            palm_pose_l_object = util.list2pose_stamped(cfg.PALM_LEFT)
            palm_pose_r_object = util.list2pose_stamped(cfg.PALM_RIGHT)

            example_args = {}
            example_args['object_pose1_world'] = object_pose1_world
            example_args['object_pose2_world'] = object_pose2_world
            example_args['palm_pose_l_object'] = palm_pose_l_object
            example_args['palm_pose_r_object'] = palm_pose_r_object
            example_args['object'] = manipulated_object
            example_args['N'] = 60  # 60
            example_args['init'] = True
            example_args['table_face'] = 0

            primitive_name = args.primitive

            mesh_file = args.config_package_path + 'descriptions/meshes/objects/' + args.object_name + '_experiments.stl'
            exp = EvalPrimitives(cfg, box_id, mesh_file)

            k = 0
            while True:
                # sample a random stable pose, and get the corresponding
                # stable orientation index
                k += 1
                # init_id = exp_running.get_rand_init()[-1]
                init_id = exp_running.get_rand_init(ind=0)[-1]

                # sample a point on the object that is valid
                # for the primitive action being executed
                point, normal, face = exp_running.sample_contact(
                    primitive_name=primitive_name)
                if point is not None:
                    break
                if k >= 10:
                    print("FAILED")
                    continue

            # get the full 6D pose palm in world, at contact location
            palm_pose_world = exp_running.get_palm_pose_world_frame(
                point,
                normal,
                primitive_name=primitive_name)

            obj_pos_world = list(p.getBasePositionAndOrientation(
                box_id,
                pb_util.PB_CLIENT)[0])
            obj_ori_world = list(p.getBasePositionAndOrientation(
                box_id,
                pb_util.PB_CLIENT)[1])

            obj_pose_world = util.list2pose_stamped(
                obj_pos_world + obj_ori_world)

            contact_obj_frame = util.convert_reference_frame(
                palm_pose_world, obj_pose_world, util.unit_pose())

            # set up inputs to the primitive planner, based on task
            # including sampled initial object pose and contacts,
            # and final object pose
            example_args['palm_pose_r_object'] = contact_obj_frame
            example_args['object_pose1_world'] = obj_pose_world

            obj_pose_final = util.list2pose_stamped(exp_running.init_poses[init_id])
            obj_pose_final.pose.position.z /= 1.155
            print("init: ")
            print(util.pose_stamped2list(object_pose1_world))
            print("final: ")
            print(util.pose_stamped2list(obj_pose_final))
            example_args['object_pose2_world'] = obj_pose_final
            example_args['table_face'] = init_id
            example_args['primitive_name'] = primitive_name
            # if trial == 0:
            #     goal_viz.update_goal_state(exp_running.init_poses[init_id])
            result = None
            try:
                result = action_planner.execute(primitive_name, example_args)
                # result = work_queue.get(block=True)
                print("reached final: " + str(result[0]))
            except ValueError:
                print("moveit failed!")
            result_queue.put(result)
            continue
        if msg == "END":
            break
        print("before sleep!")
        time.sleep(0.01)
    print("breaking")
    child_conn.close()


def calc_n(start, goal):
    dist = np.sqrt(
        (start.pose.position.x - goal.pose.position.x)**2 +
        (start.pose.position.y - goal.pose.position.y)**2
    )
    N = max(2, int(dist*100))
    return N


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

    # parent1, child1 = Pipe()
    # parent2, child2 = Pipe()
    # work_queue = Queue()
    # result_queue = Queue()
    # p1 = Process(target=worker_yumi, args=(child1, work_queue, result_queue, cfg, args))
    # p2 = Process(target=worker_yumi, args=(child2, work_queue, result_queue, cfg, args))
    # p1.start()
    # p2.start()

    # parent1.send("RESET")
    # parent2.send("RESET")

    # print("started workers")
    # time.sleep(15.0)
    # embed()

    # # setup yumi
    # data_seed = 1
    data_seed = args.np_seed
    np.random.seed(data_seed)

    # yumi_ar = Robot('yumi_palms',
    #                 pb=True,
    #                 arm_cfg={'render': args.visualize,
    #                          'self_collision': False,
    #                          'rt_simulation': True,
    #                          'seed': data_seed})

    yumi_ar = Robot('yumi_palms',
                    pb=True,
                    arm_cfg={'render': args.visualize,
                             'self_collision': False,
                             'rt_simulation': True,
                             'seed': data_seed})

    # yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    gel_id = 12

    alpha = 0.01
    K = 500
    restitution = 0.99
    dynamics_info = {}
    dynamics_info['contactDamping'] = alpha*K
    dynamics_info['contactStiffness'] = K
    dynamics_info['rollingFriction'] = args.rolling
    dynamics_info['restitution'] = restitution

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        gel_id,
        restitution=restitution,
        contactStiffness=K,
        contactDamping=alpha*K,
        rollingFriction=args.rolling
    )

    # setup yumi_gs
    # yumi_gs = YumiGelslimPybulet(yumi_ar, cfg, exec_thread=args.execute_thread)
    yumi_gs = YumiCamsGS(
        yumi_ar,
        cfg,
        exec_thread=False,
        sim_step_repeat=args.step_repeat)

    for _ in range(10):
        yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    if args.object:
        box_id = pb_util.load_urdf(
            args.config_package_path +
            'descriptions/urdf/'+args.object_name+'.urdf',
            cfg.OBJECT_POSE_3[0:3],
            cfg.OBJECT_POSE_3[3:]
        )
        # goal_obj_id = pb_util.load_urdf(
        #     args.config_package_path +
        #     'descriptions/urdf/'+args.object_name+'_trans.urdf',
        #     cfg.OBJECT_POSE_3[0:3],
        #     cfg.OBJECT_POSE_3[3:]
        # )
        # p.setCollisionFilterPair(yumi_ar.arm.robot_id, goal_obj_id, gel_id, -1, enableCollision=False)
        # p.setCollisionFilterPair(box_id, goal_obj_id, -1, -1, enableCollision=False)
        # p.setCollisionFilterPair(yumi_ar.arm.robot_id, box_id, gel_id, -1, enableCollision=True)
        # p.setCollisionFilterPair(yumi_ar.arm.robot_id, box_id, 27, -1, enableCollision=True)
                

    manipulated_object = None
    object_pose1_world = util.list2pose_stamped(cfg.OBJECT_INIT)
    object_pose2_world = util.list2pose_stamped(cfg.OBJECT_FINAL)
    palm_pose_l_object = util.list2pose_stamped(cfg.PALM_LEFT)
    palm_pose_r_object = util.list2pose_stamped(cfg.PALM_RIGHT)

    example_args = {}
    example_args['object_pose1_world'] = object_pose1_world
    example_args['object_pose2_world'] = object_pose2_world
    example_args['palm_pose_l_object'] = palm_pose_l_object
    example_args['palm_pose_r_object'] = palm_pose_r_object
    example_args['object'] = manipulated_object
    # example_args['N'] = calc_n(object_pose1_world, object_pose2_world)  # 60
    example_args['N'] = 60 # 60
    example_args['init'] = True
    example_args['table_face'] = 0

    primitive_name = args.primitive

    mesh_file = args.config_package_path + 'descriptions/meshes/objects/' + args.object_name + '_experiments.stl'
    exp_single = SingleArmPrimitives(cfg, box_id, mesh_file)
    if primitive_name == 'grasp' or primitive_name == 'pivot':
        exp_double = DualArmPrimitives(cfg, box_id, mesh_file)
        exp_running = exp_double
    else:
        exp_running = exp_single

    # setup macro_planner
    action_planner = ClosedLoopMacroActions(
        cfg,
        yumi_gs,
        box_id,
        pb_util.PB_CLIENT,
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

    cuboid_sampler = CuboidSampler('/root/catkin_ws/src/primitives/objects/cuboids/nominal_cuboid.stl')
    cuboid_fname_template = '/root/catkin_ws/src/primitives/objects/cuboids/'

    pb_util.remove_body(box_id)

    cuboid_fnames = 'test_cuboid_smaller_'
    cuboid_fname = os.path.join(cuboid_fname_template, cuboid_fnames + str(np.random.randint(4500)) + '.stl')
    obj_id, sphere_ids, mesh, goal_obj_id = cuboid_sampler.sample_cuboid_pybullet(cuboid_fname, goal=True, keypoints=False)
    p.setCollisionFilterPair(yumi_ar.arm.robot_id, goal_obj_id, gel_id, -1, enableCollision=False)
    action_planner.update_object(obj_id, mesh_file)
    exp_running.initialize_object(obj_id, cuboid_fname)    

    trans_box_lock = threading.RLock()
    goal_viz = GoalVisual(
        trans_box_lock,
        goal_obj_id,
        action_planner.pb_client,
        cfg.OBJECT_POSE_3)

    # visualize_goal_thread = threading.Thread(
    #     target=goal_viz.visualize_goal_state)
    # visualize_goal_thread.daemon = True
    # visualize_goal_thread.start()

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

    if args.debug:
        init_id = exp_running.get_rand_init(ind=2)[-1]
        obj_pose_final = util.list2pose_stamped(exp_running.init_poses[init_id])
        point, normal, face = exp_running.sample_contact(primitive_name)

        world_pose = exp_running.get_palm_pose_world_frame(
            point,
            normal,
            primitive_name=primitive_name)

        obj_pos_world = list(p.getBasePositionAndOrientation(box_id, pb_util.PB_CLIENT)[0])
        obj_ori_world = list(p.getBasePositionAndOrientation(box_id, pb_util.PB_CLIENT)[1])

        obj_pose_world = util.list2pose_stamped(obj_pos_world + obj_ori_world)
        contact_obj_frame = util.convert_reference_frame(world_pose, obj_pose_world, util.unit_pose())

        example_args['palm_pose_r_object'] = contact_obj_frame
        example_args['object_pose1_world'] = obj_pose_world

        obj_pose_final = util.list2pose_stamped(exp_running.init_poses[init_id])
        obj_pose_final.pose.position.z = obj_pose_world.pose.position.z/1.175
        print("init: ")
        print(util.pose_stamped2list(object_pose1_world))
        print("final: ")
        print(util.pose_stamped2list(obj_pose_final))
        example_args['object_pose2_world'] = obj_pose_final
        example_args['table_face'] = init_id

        plan = action_planner.get_primitive_plan(primitive_name, example_args, 'right')

        embed()

        import simulation

        for i in range(10):
            simulation.visualize_object(
                object_pose1_world,
                filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
                name="/object_initial",
                color=(1., 0., 0., 1.),
                frame_id="/yumi_body",
                scale=(1., 1., 1.))
            simulation.visualize_object(
                object_pose2_world,
                filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
                name="/object_final",
                color=(0., 0., 1., 1.),
                frame_id="/yumi_body",
                scale=(1., 1., 1.))
            rospy.sleep(.1)
        simulation.simulate(plan)
    else:
        global_start = time.time()
        face = 0
        # exp_double.reset_graph(face)
        start_time = time.time()
        success = 0

        for trial in range(args.num_trials):
            k = 0
            
            cuboid_sampler.delete_cuboid(obj_id, goal_obj_id, sphere_ids)

            cuboid_fname = os.path.join(cuboid_fname_template, cuboid_fnames + str(np.random.randint(4999)) + '.stl')    
            obj_id, sphere_ids, mesh, goal_obj_id = cuboid_sampler.sample_cuboid_pybullet(cuboid_fname, goal=True, keypoints=False)
            p.setCollisionFilterPair(yumi_ar.arm.robot_id, goal_obj_id, gel_id, -1, enableCollision=False)
            p.setCollisionFilterPair(yumi_ar.arm.robot_id, obj_id, gel_id, -1, enableCollision=True)
            p.setCollisionFilterPair(yumi_ar.arm.robot_id, obj_id, 27, -1, enableCollision=True)
            p.setCollisionFilterPair(yumi_ar.arm.robot_id, goal_obj_id, 27, -1, enableCollision=True)            
            goal_viz.update_goal_obj(goal_obj_id)
            p.changeDynamics(
                obj_id,
                -1,
                lateralFriction=0.4
            )

            action_planner.update_object(obj_id, mesh_file)
            exp_running.initialize_object(obj_id, cuboid_fname)

            while True:
                have_contact = False
                # sample a random stable pose, and get the corresponding
                # stable orientation index
                k += 1

                if primitive_name == 'pull':
                    # init_id = exp_running.get_rand_init()[-1]
                    init_id = exp_running.get_rand_init()[-1]

                    # sample a point on the object that is valid
                    # for the primitive action being executed
                    point, normal, face = exp_running.sample_contact(
                        primitive_name=primitive_name)
                    if point is not None:
                        break
                elif primitive_name == 'grasp':
                    x, y, dq, q, init_id = exp_double.get_rand_init()
                    obj_pose_world_nom = exp_double.get_obj_pose()[0]

                    palm_poses_world = exp_double.get_palm_poses_world_frame(
                        init_id,
                        obj_pose_world_nom,
                        [x, y, dq])

                    obj_pose_world = exp_double.get_obj_pose()[0]

                    if palm_poses_world is not None:
                        have_contact = True
                        break
                if k >= 10:
                    print("FAILED")
                    return

            # for _ in range(10):
            #     yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

            if primitive_name == 'pull':
                # get the full 6D pose palm in world, at contact location
                palm_pose_world = exp_running.get_palm_poses_world_frame(
                    point,
                    normal,
                    primitive_name=primitive_name)

                # get the object pose in the world frame

                # if trial == 0:
                #     parent1.send("OBJECT_POSE")
                # elif trial == 1:
                #     parent2.send("OBJECT_POSE")

                obj_pos_world = list(p.getBasePositionAndOrientation(
                    obj_id,
                    pb_util.PB_CLIENT)[0])
                obj_ori_world = list(p.getBasePositionAndOrientation(
                    obj_id,
                    pb_util.PB_CLIENT)[1])

                obj_pose_world = util.list2pose_stamped(
                    obj_pos_world + obj_ori_world)

                # obj_pose_world = work_queue.get(block=True)

                # transform the palm pose from the world frame to the object frame
                contact_obj_frame = util.convert_reference_frame(
                    palm_pose_world, obj_pose_world, util.unit_pose())

                # set up inputs to the primitive planner, based on task
                # including sampled initial object pose and contacts,
                # and final object pose
                example_args['palm_pose_r_object'] = contact_obj_frame
                example_args['object_pose1_world'] = obj_pose_world

                # obj_pose_final = util.list2pose_stamped(exp_running.init_poses[init_id])

                x, y, q, _ = exp_running.get_rand_init(execute=False, ind=init_id)
                final_nominal = exp_running.init_poses[init_id]
                final_nominal[0] = x
                final_nominal[1] = y
                final_nominal[3:] = q
                obj_pose_final = util.list2pose_stamped(final_nominal)
                goal_viz.update_goal_state(final_nominal)

                # obj_pose_final.pose.position.z /= (1.0/delta_z_height)

                obj_pose_final.pose.position.z += cfg.TABLE_HEIGHT
                # obj_pose_final.pose.position.z -= cfg.DELTA_Z

                example_args['object_pose2_world'] = obj_pose_final
                example_args['table_face'] = init_id
                example_args['primitive_name'] = primitive_name
                example_args['N'] = calc_n(obj_pose_world, obj_pose_final)
            elif primitive_name == 'grasp':
                if have_contact:
                    obj_pose_final = exp_double.goal_pose_world_frame_mod
                    palm_poses_obj_frame = {}
                    for key in palm_poses_world.keys():
                        palm_poses_obj_frame[key] = util.convert_reference_frame(
                            palm_poses_world[key], obj_pose_world, util.unit_pose()
                        )
                    example_args['palm_pose_r_object'] = palm_poses_obj_frame['right']
                    example_args['palm_pose_l_object'] = palm_poses_obj_frame['left']
                    example_args['object_pose1_world'] = obj_pose_world
                    example_args['object_pose2_world'] = obj_pose_final
                    example_args['table_face'] = init_id
                else:
                    continue

            
            try:
                # get observation (images/point cloud)
                obs, pcd = yumi_gs.get_observation(obj_id=obj_id)

                # get start/goal (obj_pose_world, obj_pose_final)
                start = util.pose_stamped2list(obj_pose_world)
                goal = util.pose_stamped2list(obj_pose_final)
                # goal_viz.update_goal_state(goal)

                # get corners (from exp? that has mesh)
                keypoints_start = np.array(exp_running.mesh_world.vertices.tolist())
                keypoints_start_homog = np.hstack(
                    (keypoints_start, np.ones((keypoints_start.shape[0], 1)))
                )
                # goal_start_frame = util.convert_reference_frame(
                #     pose_source=obj_pose_final,
                #     pose_frame_target=obj_pose_world,
                #     pose_frame_source=util.unit_pose()
                # )
                # goal_start_frame_mat = util.matrix_from_pose(goal_start_frame)
                # keypoints_goal = np.matmul(goal_start_frame_mat, keypoints_start_homog.T).T

                start_mat = util.matrix_from_pose(obj_pose_world)
                goal_mat = util.matrix_from_pose(obj_pose_final)

                T_mat = np.matmul(goal_mat, np.linalg.inv(start_mat))
                keypoints_goal = np.matmul(T_mat, keypoints_start_homog.T).T[:, :3]
                #data['keypoints_goal_corrected'] = np.matmul(
                #    T_mat, keypoints_start_homog.T).T[:, :3]
                #data['transformation_corrected'] = util.pose_stamped2list(util.pose_from_matrix(T_mat))

                contact_obj_frame_dict = {}
                contact_world_frame_dict = {}
                nearest_pt_world_dict = {}

                if primitive_name == 'pull':
                    active_arm, inactive_arm = action_planner.get_active_arm(
                        util.pose_stamped2list(obj_pose_world)
                    )

                    # get contact (palm pose object frame)
                    contact_obj_frame_dict[active_arm] = util.pose_stamped2list(contact_obj_frame)
                    contact_world_frame_dict[active_arm] = util.pose_stamped2list(palm_pose_world)
                    contact_pos = open3d.utility.DoubleVector(np.array(contact_world_frame_dict[active_arm][:3]))
                    kdtree = open3d.geometry.KDTreeFlann(pcd)
                    # nearest_pt_ind = kdtree.search_knn_vector_3d(contact_pos, 1)[1][0]
                    # nearest_pt_world_dict[active_arm] = np.asarray(pcd.points)[nearest_pt_ind]

                    contact_obj_frame_dict[inactive_arm] = None
                    contact_world_frame_dict[inactive_arm] = None
                    nearest_pt_world_dict[inactive_arm] = None
                elif primitive_name == 'grasp':
                    for key in palm_poses_obj_frame.keys():
                        contact_obj_frame_dict[key] = util.pose_stamped2list(palm_poses_obj_frame[key])
                        contact_world_frame_dict[key] = util.pose_stamped2list(palm_poses_world[key])
                        contact_pos = open3d.utility.DoubleVector(np.array(contact_world_frame[key][:3]))
                        kdtree = open3d.geometry.KDTreeFlann(pcd)
                        # nearest_pt_ind = kdtree.search_knn_vector_3d(contact_pos, 1)[1][0]
                        # nearest_pt_world_dict[key] = np.asarray(pcd.points)[nearest_pt_ind]


                # embed()

                result = action_planner.execute(primitive_name, example_args)
                if result is not None:
                    print('Trial number: ' + str(trial) + ', reached final: ' + str(result[0]))
                    print('Time so far: ' + str(time.time() - start_time))

                    if result[0]:
                        success += 1
                        sample = {}
                        sample['obs'] = obs
                        sample['start'] = start
                        sample['goal'] = goal
                        sample['keypoints_start'] = keypoints_start
                        sample['keypoints_goal'] = keypoints_goal
                        sample['transformation'] = util.pose_from_matrix(T_mat)
                        sample['contact_obj_frame'] = contact_obj_frame_dict
                        sample['contact_world_frame'] = contact_world_frame_dict
                        # sample['contact_pcd'] = nearest_pt_world_dict
                        sample['result'] = result
                        # sample['planner_args'] = example_args
                        if primitive_name == 'grasp':
                            sample['goal_face'] = exp_double.goal_face

                        if args.save_data:
                            # with open(os.path.join(pickle_path, str(trial)+'.pkl'), 'wb') as data_f:
                            #     pickle.dump(sample, data_f)
                            data_manager.save_observation(sample, str(trial))
                        print("Success: " + str(success))
                else:
                    continue

                # embed()

                # data['saved_data'].append(sample)
            #     if trial == 0:
            #         parent1.send("SAMPLE")
            #     elif trial == 1:
            #         parent2.send("SAMPLE")
            #     result = work_queue.get(block=True)

            # if trial == 0:
            #     parent1.send("SAMPLE")
            # elif trial == 1:
            #     parent2.send("SAMPLE")
            # parent1.send("SAMPLE")
            # parent2.send("SAMPLE")

            # start = time.time()
            # done = False
            # result_list = []
            # while (time.time() - start) < cfg.TIMEOUT and not done:
            #     try:
            #         result = result_queue.get(block=True)
            #         result_list.append(result)
            #         if len(result_list) == 2:
            #             done = True
            #     except result_queue.Empty:
            #         continue
            #     time.sleep(0.001)
            except ValueError as e:
                print("Value error: ")
                print(e)

            # time.sleep(1.0)

            # pose = util.pose_stamped2list(yumi_gs.compute_fk(yumi_gs.get_jpos(arm='right')))
            # pos, ori = pose[:3], pose[3:]

            # # pose = yumi_gs.get_ee_pose()
            # # pos, ori = pose[0], pose[1]
            # # pos[2] -= 0.0714
            # pos[2] += 0.001
            # r_jnts = yumi_gs.compute_ik(pos, ori, yumi_gs.get_jpos(arm='right'))
            # l_jnts = yumi_gs.get_jpos(arm='left')

            # if r_jnts is not None:
            #     for _ in range(10):
            #         pos[2] += 0.001
            #         r_jnts = yumi_gs.compute_ik(pos, ori, yumi_gs.get_jpos(arm='right'))
            #         l_jnts = yumi_gs.get_jpos(arm='left')

            #         if r_jnts is not None:
            #             yumi_gs.update_joints(list(r_jnts) + l_jnts)
            #         time.sleep(0.1)            

            time.sleep(0.1)
            for _ in range(10):
                yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)
            
            # cuboid_sampler.delete_cuboid(obj_id, [])

            # for _ in range(10):
            #     j_pos = cfg.RIGHT_INIT + cfg.LEFT_INIT
            #     for ind, jnt_id in enumerate(yumi_ar.arm.arm_jnt_ids):
            #         p.resetJointState(
            #             yumi_ar.arm.robot_id,
            #             jnt_id,
            #             targetValue=j_pos[ind]
            #         )

            # yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

            # p.resetJointStatesMultiDof(
            #     yumi_ar.arm.robot_id,
            #     yumi_ar.arm.arm_jnt_ids,
            #     targetValues=j_pos)
            # parent1.send("HOME")
            # parent2.send("HOME")

            # time.sleep(1.0)

            # embed()

    # embed()

    print("TOTAL TIME: " + str(time.time() - global_start))


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

    args = parser.parse_args()
    main(args)
