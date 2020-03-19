import os, sys
import time
import argparse
import numpy as np
from multiprocessing import Process, Pipe, Queue, Manager
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
from macro_actions import ClosedLoopMacroActions, YumiGelslimPybulet
from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

from yacs.config import CfgNode as CN
from closed_loop_experiments_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler
from data_gen_utils import YumiCamsGS, DataManager, MultiBlockManager, GoalVisual


def worker_yumi(child_conn, work_queue, result_queue,
                cfg, args, seed, worker_id,
                global_info_dict, worker_flag_dict,
                planning_lock):
    while True:
        try:
            if not child_conn.poll(0.0001):
                continue
            msg = child_conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if msg == "RESET":
            data_seed = seed
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

            # goal_obj_id = yumi_ar.pb_client.load_urdf(
            #     args.config_package_path +
            #     'descriptions/urdf/'+args.object_name+'_trans.urdf',
            #     cfg.OBJECT_POSE_3[0:3],
            #     cfg.OBJECT_POSE_3[3:]
            # )
            # p.setCollisionFilterPair(yumi_ar.arm.robot_id, goal_obj_id, r_gel_id, -1, enableCollision=False)
            # p.setCollisionFilterPair(obj_id, goal_obj_id, -1, -1, enableCollision=False)
            p.setCollisionFilterPair(yumi_ar.arm.robot_id, obj_id, r_gel_id, -1, enableCollision=True)
            p.setCollisionFilterPair(yumi_ar.arm.robot_id, obj_id, 27, -1, enableCollision=True)

            yumi_ar.pb_client.reset_body(
                obj_id,
                cfg.OBJECT_POSE_3[:3],
                cfg.OBJECT_POSE_3[3:])

            # yumi_ar.pb_client.reset_body(
            #     goal_obj_id,
            #     cfg.OBJECT_POSE_3[:3],
            #     cfg.OBJECT_POSE_3[3:])

            primitive_name = args.primitive
            mesh_file = '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids/realsense_box_experiments.stl'
            cuboid_fname = mesh_file
            face = 0

            # setup macro_planner
            action_planner = ClosedLoopMacroActions(
                cfg,
                yumi_gs,
                obj_id,
                yumi_ar.pb_client.get_client_id(),
                args.config_package_path,
                object_mesh_file=mesh_file,
                replan=args.replan
            )

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

            if primitive_name == 'grasp' or primitive_name == 'pivot':
                exp_running = exp_double
            else:
                exp_running = exp_single

            action_planner.update_object(obj_id, mesh_file)
            exp_single.initialize_object(obj_id, cuboid_fname)
            exp_double.initialize_object(obj_id, cuboid_fname, face)

            # goal_viz = GoalVisual(
            #     trans_box_lock,
            #     goal_obj_id,
            #     action_planner.pb_client,
            #     cfg.OBJECT_POSE_3)

            pickle_path = os.path.join(
                args.data_dir,
                primitive_name,
                args.experiment_name
            )

            if not os.path.exists(pickle_path):
                os.makedirs(pickle_path)

            data_manager = DataManager(pickle_path)
            continue
        if msg == "HOME":
            yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)
            continue
        if msg == "OBJECT_POSE":
            continue
        if msg == "SAMPLE":
            global_info_dict['trial'] += 1
            # print('Success: ' + str(global_info_dict['success']) +
            #         ' Trial number: ' + str(global_info_dict['trial']) +
            #         ' Worker ID: ' + str(worker_id))

            worker_flag_dict[worker_id] = False

            success = False
            start_face = 1
            plan_args = exp_running.get_random_primitive_args(
                ind=start_face,
                random_goal=False,
                execute=True
            )
            try:
                obs, pcd = yumi_gs.get_observation(obj_id=obj_id)

                obj_pose_world = plan_args['object_pose1_world']
                obj_pose_final = plan_args['object_pose2_world']

                contact_obj_frame, contact_world_frame = {}, {}

                contact_obj_frame['right'] = plan_args['palm_pose_r_object']
                contact_world_frame['right'] = plan_args['palm_pose_r_world']
                contact_obj_frame['left'] = plan_args['palm_pose_l_object']
                contact_world_frame['left'] = plan_args['palm_pose_l_world']

                start = util.pose_stamped2list(obj_pose_world)
                goal = util.pose_stamped2list(obj_pose_final)

                keypoints_start = np.array(exp_running.mesh_world.vertices.tolist())
                keypoints_start_homog = np.hstack(
                    (keypoints_start, np.ones((keypoints_start.shape[0], 1)))
                )

                start_mat = util.matrix_from_pose(obj_pose_world)
                goal_mat = util.matrix_from_pose(obj_pose_final)

                T_mat = np.matmul(goal_mat, np.linalg.inv(start_mat))
                keypoints_goal = np.matmul(T_mat, keypoints_start_homog.T).T[:, :3]

                contact_obj_frame_dict = {}
                contact_world_frame_dict = {}
                nearest_pt_world_dict = {}
                corner_norms = {}
                down_pcd_norms = {}

                if primitive_name == 'pull':
                    # active_arm, inactive_arm = action_planner.get_active_arm(
                    #     util.pose_stamped2list(obj_pose_world)
                    # )
                    active_arm = action_planner.active_arm
                    inactive_arm = action_planner.inactive_arm

                    contact_obj_frame_dict[active_arm] = util.pose_stamped2list(contact_obj_frame[active_arm])
                    contact_world_frame_dict[active_arm] = util.pose_stamped2list(contact_world_frame[active_arm])
                    # contact_pos = open3d.utility.DoubleVector(np.array(contact_world_frame_dict[active_arm][:3]))
                    # kdtree = open3d.geometry.KDTreeFlann(pcd)
                    # nearest_pt_ind = kdtree.search_knn_vector_3d(contact_pos, 1)[1][0]
                    # nearest_pt_world_dict[active_arm] = np.asarray(pcd.points)[nearest_pt_ind]
                    contact_pos = np.array(contact_world_frame_dict[active_arm][:3])

                    corner_dists = (np.asarray(keypoints_start) - contact_pos)
                    corner_norms[active_arm] = np.linalg.norm(corner_dists, axis=1)

                    down_pcd_dists = (obs['down_pcd_pts'] - contact_pos)
                    down_pcd_norms[active_arm] = np.linalg.norm(down_pcd_dists, axis=1)

                    contact_obj_frame_dict[inactive_arm] = None
                    contact_world_frame_dict[inactive_arm] = None
                    nearest_pt_world_dict[inactive_arm] = None
                    corner_norms[inactive_arm] = None
                    down_pcd_norms[inactive_arm] = None
                elif primitive_name == 'grasp':
                    for key in contact_obj_frame.keys():
                        contact_obj_frame_dict[key] = util.pose_stamped2list(contact_obj_frame[key])
                        contact_world_frame_dict[key] = util.pose_stamped2list(contact_world_frame[key])

                        contact_pos = np.array(contact_world_frame_dict[key][:3])

                        corner_dists = (np.asarray(keypoints_start) - contact_pos)
                        corner_norms[key] = np.linalg.norm(corner_dists, axis=1)

                        down_pcd_dists = (obs['down_pcd_pts'] - contact_pos)
                        down_pcd_norms[key] = np.linalg.norm(down_pcd_dists, axis=1)

                        # contact_pos = open3d.utility.DoubleVector(np.array(contact_world_frame_dict[key][:3]))
                        # kdtree = open3d.geometry.KDTreeFlann(pcd)
                        # nearest_pt_ind = kdtree.search_knn_vector_3d(contact_pos, 1)[1][0]
                        # nearest_pt_world_dict[key] = np.asarray(pcd.points)[nearest_pt_ind]

                print('Worker ID: %d, Waiting to plan...' % worker_id)
                while True:
                    time.sleep(0.01)
                    work_queue_empty = work_queue.empty()
                    if work_queue_empty:
                        planner_available = True
                        break
                if planner_available:
                    print('Worker ID: %d, Planning!...' % worker_id)
                    work_queue.put(True)
                    time.sleep(1.0)                
                    result = action_planner.execute(primitive_name, plan_args)
                    time.sleep(1.0)
                    while not work_queue.empty():
                        work_queue.get()

                if result is not None:
                    if result[0]:
                        success = True
                        global_info_dict['success'] += 1
                        sample = {}
                        sample['obs'] = obs
                        sample['start'] = start
                        sample['goal'] = goal
                        sample['keypoints_start'] = keypoints_start
                        sample['keypoints_goal'] = keypoints_goal
                        sample['keypoint_dists'] = corner_norms
                        sample['down_pcd_pts'] = obs['down_pcd_pts']
                        sample['down_pcd_dists'] = down_pcd_norms
                        sample['transformation'] = util.pose_from_matrix(T_mat)
                        sample['contact_obj_frame'] = contact_obj_frame_dict
                        sample['contact_world_frame'] = contact_world_frame_dict
                        # sample['contact_pcd'] = nearest_pt_world_dict
                        sample['result'] = result
                        if primitive_name == 'grasp':
                            sample['goal_face'] = exp_double.goal_face

                        if args.save_data:
                            data_manager.save_observation(sample, str(global_info_dict['trial']))
                        # print("Success: " + str(global_info_dict['success']))
            except ValueError as e:
                print("Value error: ")
                print(e)
                result = None
                time.sleep(1.0)
                while not work_queue.empty():
                    work_queue.get()                
            worker_result = {}
            worker_result['result'] = result
            worker_result['id'] = worker_id
            worker_result['success'] = success
            result_queue.put(worker_result)
            child_conn.send('DONE')
            continue
        if msg == "END":
            break
        time.sleep(0.001)
    print('Breaking Worker ID: ' + str(worker_id))
    child_conn.close()


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


class WorkerManager(object):
    def __init__(self, cfg, args, save_path,
                 work_queue, result_queue, global_manager):
        self.cfg = cfg
        self.args = args
        self.save_path = save_path

        self.work_queue = work_queue
        self.result_queue = result_queue
        self.global_manager = global_manager
        self.planning_lock = threading.Lock()

        self.global_info_dict = self.global_manager.dict()
        self.global_info_dict['trial'] = 0
        self.global_info_dict['success'] = 0
        self.worker_flag_dict = self.global_manager.dict()

    def setup_workers(self, num_workers):
        """Setup function to instantiate the desired number of
        workers. Pipes and Processes set up, stored internally,
        and started.

        Args:
            num_workers (int): Desired number of worker processes
        """
        worker_ids = np.arange(num_workers, dtype=np.int64).tolist()
        seeds = np.arange(self.args.np_seed, self.args.np_seed + num_workers, dtype=np.int64).tolist()

        self._worker_ids = worker_ids
        self.seeds = seeds

        self._pipes = {}
        self._processes = {}
        for i, worker_id in enumerate(self._worker_ids):
            parent, child = Pipe(duplex=True)
            self.worker_flag_dict[worker_id] = True
            p = Process(
                target=worker_yumi,
                args=(
                    child,
                    self.work_queue,
                    self.result_queue,
                    self.cfg,
                    self.args,
                    seeds[i],
                    worker_id,
                    self.global_info_dict,
                    self.worker_flag_dict,
                    self.planning_lock
                )
            )
            pipe = {}
            pipe['parent'] = parent
            pipe['child'] = child

            self._pipes[worker_id] = pipe
            self._processes[worker_id] = p

        for i, worker_id in enumerate(self._worker_ids):
            self._processes[worker_id].start()
            self._pipes[worker_id]['parent'].send('RESET')
            print('RESET WORKER ID: ' + str(worker_id))
        print('FINISHED WORKER SETUP')

    def sample_trials(self, total_num_trials):
        num_trials = self.get_trial_number()
        while num_trials < total_num_trials:
            num_trials = self.get_trial_number()
            for i, worker_id in enumerate(self._worker_ids):
                if self.get_worker_ready(worker_id):
                    self._pipes[worker_id]['parent'].send('SAMPLE')
                    self.worker_flag_dict[worker_id] = False
                    # self.work_queue.put(True)
                    print('Trial: ' + str(num_trials) +
                          ' Success: ' + str(self.get_success_number()))
            time.sleep(0.01)
        print('\n\n\n\nDone!\n\n\n\n')

    def yumi_worker_home_target(self):
        while True:
            try:
                for i, worker_id in enumerate(self._worker_ids):
                    parent_conn = self._pipes[worker_id]['parent']
                    try:
                        if not parent_conn.poll(0.0001):
                            continue
                        msg = parent_conn.recv()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if msg == 'DONE':
                        parent_conn.send('HOME')
                        self.worker_flag_dict[worker_id] = True
                        continue
            except KeyboardInterrupt:
                break
            time.sleep(0.001)

    def result_buffer_target(self):
        while True:
            try:
                if not self.result_queue.empty():
                    try:
                        result = self.result_queue.get(timeout=1.0)
                        # print('Got result!')
                        # for key in result.keys():
                        #     print(key, result[key])
                    except Exception as e:
                        print('Could not get result')
                        print(e)
            except (EOFError, KeyboardInterrupt):
                break
            time.sleep(0.001)

    def get_pipes(self):
        return self._pipes

    def get_processes(self):
        return self._processes

    def get_worker_ids(self):
        return self._worker_ids

    def get_worker_ready(self, worker_id):
        return self.worker_flag_dict[worker_id]

    def get_global_info_dict(self):
        """Returns the globally shared dictionary of data
        generation information, including success rate and
        trial number

        Returns:
            dict: Dictionary of global information shared
                between workers
        """
        return self.global_info_dict

    def get_trial_number(self):
        return self.global_info_dict['trial']

    def get_success_number(self):
        return self.global_info_dict['success']


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

    manager = Manager()
    work_queue = Queue()
    result_queue = Queue()

    primitive_name = args.primitive

    pickle_path = os.path.join(
        args.data_dir,
        primitive_name,
        args.experiment_name
    )

    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)

    yumi_worker_manager = WorkerManager(
        cfg, args, pickle_path,
        work_queue, result_queue, manager
    )

    yumi_worker_manager.setup_workers(num_workers=1)

    result_thread = threading.Thread(
        target=yumi_worker_manager.result_buffer_target)
    result_thread.daemon = True
    result_thread.start()

    home_thread = threading.Thread(
        target=yumi_worker_manager.yumi_worker_home_target)
    home_thread.daemon = True
    home_thread.start()

    print('CALLING TRIAL SAMPLING')
    yumi_worker_manager.sample_trials(total_num_trials=100)

    embed()

    plan = action_planner.get_primitive_plan(primitive_name, example_args, 'right')

    embed()

    import simulation

    for i in range(10):
        simulation.visualize_object(
            object_pose1_world,
            filepath="package://config/descriptions/meshes/objects/cuboids/" + cuboid_fname.split('objects/cuboids')[1],
            name="/object_initial",
            color=(1., 0., 0., 1.),
            frame_id="/yumi_body",
            scale=(1., 1., 1.))
        simulation.visualize_object(
            object_pose2_world,
            filepath="package://config/descriptions/meshes/objects/cuboids/" + cuboid_fname.split('objects/cuboids')[1],
            name="/object_final",
            color=(0., 0., 1., 1.),
            frame_id="/yumi_body",
            scale=(1., 1., 1.))
        rospy.sleep(.1)
    simulation.simulate(plan, cuboid_fname.split('objects/cuboids')[1])


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
