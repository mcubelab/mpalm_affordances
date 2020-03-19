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
from macro_actions import ClosedLoopMacroActions, YumiGelslimPybulet
from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

from yacs.config import CfgNode as CN
from closed_loop_experiments_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler
from data_gen_utils import YumiCamsGS, DataManager, MultiBlockManager, GoalVisual


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


# class WorkerManager(object):
#     def __init__(self, cfg, args, save_path):

#     def setup_workers(self, num_workers):

#     def worker_yumi(self, child_conn, work_queue, result_queue
#                     seed):

def worker_yumi(child_conn, work_queue, result_queue,
                cfg, args, seed, save_path):
    while True:
        # print("here!")
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
            start_face = 1
            plan_args = exp_running.get_random_primitive_args(
                ind=start_face,
                random_goal=True,
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

                result = action_planner.execute(primitive_name, plan_args)
                if result is not None:
                    if result[0]:
                        success += 1
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
                            data_manager.save_observation(sample, str(trial))
                        print("Success: " + str(success))
            except ValueError as e:
                print("Value error: ")
                print(e)
            result_queue.put(result)
            continue
        if msg == "END":
            break
        print("before sleep!")
        time.sleep(0.01)
    print("breaking")
    child_conn.close()


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

    parent1, child1 = Pipe()
    parent2, child2 = Pipe()
    work_queue = Queue()
    result_queue = Queue()
    p1 = Process(target=worker_yumi, args=(child1, work_queue, result_queue, cfg, args, args.np_seed))
    p2 = Process(target=worker_yumi, args=(child2, work_queue, result_queue, cfg, args, args.np_seed+1))
    p1.start()
    p2.start()

    parent1.send("RESET")
    parent2.send("RESET")

    print("started workers")
    time.sleep(15.0)

    parent1.send("SAMPLE")
    parent2.send("SAMPLE")
    result1 = result_queue.get(block=True)
    result2 = result_queue.get(block=True)
    print('Result (1): ')
    print(result1)
    print('Result (2): ')
    print(result2)

    embed()

    # # setup yumi
    # data_seed = 1
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

    if args.object:
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
    example_args['N'] = 60
    example_args['init'] = True
    example_args['table_face'] = 0

    primitive_name = args.primitive

    mesh_file = args.config_package_path + 'descriptions/meshes/objects/' + args.object_name + '_experiments.stl'
    exp_single = SingleArmPrimitives(
        cfg,
        yumi_ar.pb_client.get_client_id(),
        obj_id,
        mesh_file)
    if primitive_name == 'grasp' or primitive_name == 'pivot':
        exp_double = DualArmPrimitives(
            cfg,
            yumi_ar.pb_client.get_client_id(),
            obj_id,
            mesh_file)
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

    if args.debug:
        if args.multi:
            cuboid_sampler.delete_cuboid(obj_id, goal_obj_id, sphere_ids)

            cuboid_fname = cuboid_manager.get_cuboid_fname()
            obj_id, sphere_ids, mesh, goal_obj_id = cuboid_sampler.sample_cuboid_pybullet(
                cuboid_fname,
                goal=True,
                keypoints=False)

            cuboid_manager.filter_collisions(obj_id, goal_obj_id)
            goal_viz.update_goal_obj(goal_obj_id)
            p.changeDynamics(
                obj_id,
                -1,
                lateralFriction=0.4
            )

            action_planner.update_object(obj_id, mesh_file)
            exp_running.initialize_object(obj_id, cuboid_fname)
            print('Reset multi block!')
        else:
            cuboid_fname = '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids/realsense_box_experiments.stl'

        for _ in range(args.num_obj_samples):
            if primitive_name == 'pull':
                init_id = exp_running.get_rand_init(ind=2)[-1]
                obj_pose_final = util.list2pose_stamped(exp_running.init_poses[init_id])
                point, normal, face = exp_running.sample_contact(primitive_name)

                world_pose = exp_running.get_palm_poses_world_frame(
                    point,
                    normal,
                    primitive_name=primitive_name)

                obj_pos_world = list(p.getBasePositionAndOrientation(
                    obj_id, yumi_ar.pb_client.get_client_id())[0])
                obj_ori_world = list(p.getBasePositionAndOrientation(
                    obj_id, yumi_ar.pb_client.get_client_id())[1])

                obj_pose_world = util.list2pose_stamped(obj_pos_world + obj_ori_world)
                contact_obj_frame = util.convert_reference_frame(
                    world_pose, obj_pose_world, util.unit_pose())

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
            elif primitive_name == 'grasp':
                k = 0
                have_contact = False
                contact_face = None
                while True:
                    x, y, dq, q, init_id = exp_running.get_rand_init()
                    obj_pose_world_nom = exp_running.get_obj_pose()[0]

                    palm_poses_world = exp_running.get_palm_poses_world_frame(
                        init_id,
                        obj_pose_world_nom,
                        [x, y, dq])

                    # get_palm_poses_world_frame may adjust the
                    # initial object pose, so need to check it again
                    obj_pose_world = exp_running.get_obj_pose()[0]

                    if palm_poses_world is not None:
                        have_contact = True
                        break
                    k += 1
                    if k >= 10:
                        print("FAILED")
                        break

                if have_contact:
                    obj_pose_final = exp_running.goal_pose_world_frame_mod
                    palm_poses_obj_frame = {}
                    for key in palm_poses_world.keys():
                        palm_poses_obj_frame[key] = util.convert_reference_frame(
                            palm_poses_world[key], obj_pose_world, util.unit_pose())

                    example_args['palm_pose_r_object'] = palm_poses_obj_frame['right']
                    example_args['palm_pose_l_object'] = palm_poses_obj_frame['left']
                    example_args['object_pose1_world'] = obj_pose_world
                    example_args['object_pose2_world'] = obj_pose_final
                    example_args['table_face'] = init_id

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
    else:
        global_start = time.time()
        face = 0
        # exp_double.reset_graph(face)
        start_time = time.time()
        success = 0

        for trial in range(args.num_trials):
            k = 0

            if args.multi:
                cuboid_sampler.delete_cuboid(obj_id, goal_obj_id, sphere_ids)

                cuboid_fname = cuboid_manager.get_cuboid_fname()
                obj_id, sphere_ids, mesh, goal_obj_id = cuboid_sampler.sample_cuboid_pybullet(
                    cuboid_fname,
                    goal=True,
                    keypoints=False)

                cuboid_manager.filter_collisions(obj_id, goal_obj_id)
                goal_viz.update_goal_obj(goal_obj_id)
                p.changeDynamics(
                    obj_id,
                    -1,
                    lateralFriction=0.4
                )

                action_planner.update_object(obj_id, mesh_file)
                exp_running.initialize_object(obj_id, cuboid_fname)
                print('Reset multi block!')

            for _ in range(args.num_obj_samples):

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
                        yumi_ar.pb_client.get_client_id())[0])
                    obj_ori_world = list(p.getBasePositionAndOrientation(
                        obj_id,
                        yumi_ar.pb_client.get_client_id())[1])

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
                    obj_pose_final.pose.position.z += cfg.TABLE_HEIGHT

                    example_args['object_pose2_world'] = obj_pose_final
                    example_args['table_face'] = init_id
                    example_args['primitive_name'] = primitive_name
                    example_args['N'] = exp_running.calc_n(
                        obj_pose_world, obj_pose_final)
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
                    obs, pcd = yumi_gs.get_observation(obj_id=obj_id)

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
                            contact_pos = open3d.utility.DoubleVector(np.array(contact_world_frame_dict[key][:3]))
                            kdtree = open3d.geometry.KDTreeFlann(pcd)
                            # nearest_pt_ind = kdtree.search_knn_vector_3d(contact_pos, 1)[1][0]
                            # nearest_pt_world_dict[key] = np.asarray(pcd.points)[nearest_pt_ind]

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
                            if primitive_name == 'grasp':
                                sample['goal_face'] = exp_double.goal_face

                            if args.save_data:
                                data_manager.save_observation(sample, str(trial))
                            print("Success: " + str(success))
                    else:
                        continue

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

    parser.add_argument(
        '--multi', action='store_true'
    )

    parser.add_argument(
        '--num_obj_samples', type=int, default=10
    )

    args = parser.parse_args()
    main(args)
