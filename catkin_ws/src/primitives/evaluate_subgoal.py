import os
import os.path as osp
import sys
import time
import argparse
import numpy as np
import rospy
import signal
import threading
from multiprocessing import Pipe, Queue
import pickle
import open3d
import copy
from random import shuffle
from IPython import embed

from airobot import Robot
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions
from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

from multistep_planning_eval_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler
from data_gen_utils import YumiCamsGS, DataManager, MultiBlockManager, GoalVisual
import simulation
from helper import registration as reg
from eval_utils.visualization_tools import PCDVis, PalmVis
from eval_utils.experiment_recorder import GraspEvalManager
from helper.pointcloud_planning import (
    PointCloudNode, 
    GraspSamplerVAEPubSub, PullSamplerVAEPubSub,
    GraspSamplerTransVAEPubSub,
    GraspSamplerBasic, PullSamplerBasic)
from planning import grasp_planning_wf, pulling_planning_wf


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


def main(args):
    # get configuration
    cfg_file = osp.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    rospy.init_node('EvalSubgoal')
    signal.signal(signal.SIGINT, signal_handler)

    # setup data saving paths
    data_seed = args.np_seed
    primitive_name = args.primitive


    problems_file = '/root/catkin_ws/src/primitives/data/planning/test_problems_0/demo_0.pkl'
    with open(problems_file, 'rb') as f:
        problems_data = pickle.load(f)

    prob_inds = np.arange(len(problems_data), dtype=np.int64).tolist()
    data_inds = np.arange(len(problems_data[0]['problems']), dtype=np.int64).tolist()

    pickle_path = osp.join(
        args.data_dir,
        primitive_name,
        args.experiment_name
    )

    if args.save_data:
        suf_i = 0
        original_pickle_path = pickle_path
        # while True:
        #     if osp.exists(pickle_path):
        #         suffix = '_%d' % suf_i
        #         pickle_path = original_pickle_path + suffix
        #         suf_i += 1
        #         data_seed += 1
        #     else:
        #         os.makedirs(pickle_path)
        #         break

        if not osp.exists(pickle_path):
            os.makedirs(pickle_path)

    np.random.seed(data_seed)

    # initialize airobot and modify dynamics
    yumi_ar = Robot('yumi_palms',
                    pb=True,
                    pb_cfg={'gui': args.visualize,
                            'opengl_render': False},
                    arm_cfg={'self_collision': False,
                             'seed': data_seed})

    r_gel_id = cfg.RIGHT_GEL_ID
    l_gel_id = cfg.LEFT_GEL_ID
    table_id = cfg.TABLE_ID

    alpha = cfg.ALPHA
    K = cfg.GEL_CONTACT_STIFFNESS
    restitution = cfg.GEL_RESTITUION

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        r_gel_id,
        restitution=restitution,
        contactStiffness=K,
        contactDamping=alpha*K,
        rollingFriction=args.rolling,
        lateralFriction=0.5
    )

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        l_gel_id,
        restitution=restitution,
        contactStiffness=K,
        contactDamping=alpha*K,
        rollingFriction=args.rolling,
        lateralFriction=0.5
    )

    # initialize PyBullet + MoveIt! + ROS yumi interface
    yumi_gs = YumiCamsGS(
        yumi_ar,
        cfg,
        exec_thread=False,
        sim_step_repeat=args.sim_step_repeat
    )

    yumi_ar.arm.go_home(ignore_physics=True)

    # initialize object sampler
    cuboid_sampler = CuboidSampler(
        osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/config/descriptions/meshes/objects/cuboids/nominal_cuboid.stl'),
        pb_client=yumi_ar.pb_client)
    cuboid_fname_template = osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/config/descriptions/meshes/objects/cuboids/')

    cuboid_manager = MultiBlockManager(
        cuboid_fname_template,
        cuboid_sampler,
        robot_id=yumi_ar.arm.robot_id,
        table_id=table_id,
        r_gel_id=r_gel_id,
        l_gel_id=l_gel_id)

    if args.multi:
        # cuboid_fname = cuboid_manager.get_cuboid_fname()
        # cuboid_fname = str(osp.join(
        #     '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids',
        #     problems_data[0]['object_name']))

        # get object name
        k = 0
        prob_inds = copy.deepcopy(list(np.arange(len(problems_data), dtype=np.int64)))
        shuffle(prob_inds)
        while True:
            if len(prob_inds) == 0:
                print('Done with test problems!')
                return
            prob_ind = prob_inds.pop()
            obj_name = problems_data[prob_ind]['object_name'].split('.stl')[0]
            if osp.exists(osp.join(pickle_path, obj_name)):
                continue
            os.makedirs(osp.join(pickle_path, obj_name))
            break
        cuboid_fname = str(osp.join(
            '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids',
            obj_name + '.stl'))

    else:
        cuboid_fname = args.config_package_path + 'descriptions/meshes/objects/' + \
            args.object_name + '.stl'
    mesh_file = cuboid_fname
    print("Cuboid file: " + cuboid_fname)

    if args.goal_viz:
        goal_visualization = True
    else:
        goal_visualization = False

    obj_id, sphere_ids, mesh, goal_obj_id = \
        cuboid_sampler.sample_cuboid_pybullet(
            cuboid_fname,
            goal=goal_visualization,
            keypoints=False)

    cuboid_manager.filter_collisions(obj_id, goal_obj_id)

    p.changeDynamics(
        obj_id,
        -1,
        lateralFriction=1.0
    )

    goal_faces = [0, 1, 2, 3, 4, 5]
    # shuffle(goal_faces)
    goal_face = goal_faces[0]

    # initialize primitive args samplers
    exp_single = SingleArmPrimitives(
        cfg,
        yumi_ar.pb_client.get_client_id(),
        obj_id,
        cuboid_fname)
    k = 0
    while True:
        k += 1
        if k > 10:
            print('FAILED TO BUILD GRASPING GRAPH')
            return
        try:
            exp_double = DualArmPrimitives(
                cfg,
                yumi_ar.pb_client.get_client_id(),
                obj_id,
                cuboid_fname,
                goal_face=goal_face)
            break
        except ValueError as e:
            print(e)
            yumi_ar.pb_client.remove_body(obj_id)
            if goal_visualization:
                yumi_ar.pb_client.remove_body(goal_obj_id)
            cuboid_fname = cuboid_manager.get_cuboid_fname()
            print("Cuboid file: " + cuboid_fname)

            obj_id, sphere_ids, mesh, goal_obj_id = \
                cuboid_sampler.sample_cuboid_pybullet(
                    cuboid_fname,
                    goal=goal_visualization,
                    keypoints=False)

            cuboid_manager.filter_collisions(obj_id, goal_obj_id)

            p.changeDynamics(
                obj_id,
                -1,
                lateralFriction=1.0)
    if primitive_name == 'grasp':
        exp_running = exp_double
    else:
        exp_running = exp_single

    # initialize macro action interface
    action_planner = ClosedLoopMacroActions(
        cfg,
        yumi_gs,
        obj_id,
        yumi_ar.pb_client.get_client_id(),
        args.config_package_path,
        replan=args.replan,
        object_mesh_file=mesh_file
    )

    if goal_visualization:
        trans_box_lock = threading.RLock()
        goal_viz = GoalVisual(
            trans_box_lock,
            goal_obj_id,
            action_planner.pb_client,
            cfg.OBJECT_POSE_3)

    action_planner.update_object(obj_id, mesh_file)
    exp_single.initialize_object(obj_id, cuboid_fname)

    # prep save info
    dynamics_info = {}
    dynamics_info['contactDamping'] = alpha*K
    dynamics_info['contactStiffness'] = K
    dynamics_info['rollingFriction'] = args.rolling
    dynamics_info['restitution'] = restitution

    data = {}
    data['saved_data'] = []
    data['metadata'] = {}
    data['metadata']['mesh_file'] = mesh_file
    data['metadata']['cfg'] = cfg
    data['metadata']['dynamics'] = dynamics_info
    data['metadata']['cam_cfg'] = yumi_gs.cam_setup_cfg
    data['metadata']['step_repeat'] = args.sim_step_repeat
    data['metadata']['seed'] = data_seed
    data['metadata']['seed_original'] = args.np_seed

    metadata = data['metadata']

    data_manager = DataManager(pickle_path)
    # pred_dir = osp.join(os.environ['CODE_BASE'], cfg.PREDICTION_DIR)
    # obs_dir = osp.join(os.environ['CODE_BASE'], cfg.OBSERVATION_DIR)
    pred_dir = cfg.PREDICTION_DIR
    obs_dir = cfg.OBSERVATION_DIR
    if not osp.exists(pred_dir):
        os.makedirs(pred_dir)
    if not osp.exists(obs_dir):
        os.makedirs(obs_dir)

    if args.save_data:
        with open(osp.join(pickle_path, 'metadata.pkl'), 'wb') as mdata_f:
            pickle.dump(metadata, mdata_f)

    # prep visualization tools
    palm_mesh_file = osp.join(os.environ['CODE_BASE'],
                              cfg.PALM_MESH_FILE)
    table_mesh_file = osp.join(os.environ['CODE_BASE'],
                               cfg.TABLE_MESH_FILE)
    viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
    viz_pcd = PCDVis()

    # pull_sampler = PullSamplerBasic()
    pull_sampler = PullSamplerVAEPubSub(
        obs_dir=obs_dir,
        pred_dir=pred_dir
    )    
    grasp_sampler = GraspSamplerVAEPubSub(
        default_target=None,
        obs_dir=obs_dir,
        pred_dir=pred_dir,
        pointnet=args.pointnet
    )
    # grasp_sampler = GraspSamplerTransVAEPubSub(
    #     None,
    #     obs_dir,
    #     pred_dir,
    #     pointnet=args.pointnet
    # )
    # grasp_sampler = GraspSamplerBasic(
    #     default_target=None
    # )   
    parent, child = Pipe(duplex=True)
    work_queue, result_queue = Queue(), Queue()

    experiment_manager = GraspEvalManager(
        yumi_gs,
        yumi_ar.pb_client.get_client_id(),
        pickle_path,
        args.exp_name,
        parent,
        child,
        work_queue,
        result_queue,
        cfg
    )

    experiment_manager.set_object_id(
        obj_id,
        cuboid_fname
    )

    # begin runs
    total_trials = 0
    total_executions = 0
    total_face_success = 0

    # for _ in range(args.num_blocks):
    for problem_ind in range(1, len(problems_data)):
        for goal_face in goal_faces:
            try:
                exp_double.initialize_object(obj_id, cuboid_fname, goal_face)
            except ValueError as e:
                print('Goal face: ' + str(goal_face), e)
                continue
            for _ in range(args.num_obj_samples):
                yumi_ar.arm.go_home(ignore_physics=True)
                obj_data = experiment_manager.get_object_data()
                if obj_data['trials'] > 0:
                    kvs = {}
                    kvs['trials'] = obj_data['trials']
                    kvs['grasp_success'] = obj_data['grasp_success'] * 100.0 / obj_data['trials']
                    kvs['mp_success'] = obj_data['mp_success'] * 100.0 / obj_data['trials']
                    kvs['face_success'] = obj_data['face_success'] * 100.0 / obj_data['trials']
                    kvs['pos_err'] = np.mean(obj_data['final_pos_error'])
                    kvs['ori_err'] = np.mean(obj_data['final_ori_error'])
                    string = ''

                    for k, v in kvs.items():
                        string += "%s: %.3f, " % (k,v)
                    print(string)

                total_trials += 1
                if primitive_name == 'grasp':
                    start_face = exp_double.get_valid_ind()
                    if start_face is None:
                        print('Could not find valid start face')
                        continue
                    plan_args = exp_double.get_random_primitive_args(ind=start_face,
                                                                     random_goal=True,
                                                                     execute=True)
                elif primitive_name == 'pull':
                    plan_args = exp_single.get_random_primitive_args(ind=goal_face,
                                                                     random_goal=True,
                                                                     execute=True)

                start_pose = plan_args['object_pose1_world']
                goal_pose = plan_args['object_pose2_world']

                if goal_visualization:
                    goal_viz.update_goal_state(util.pose_stamped2list(goal_pose))
                    goal_viz.hide_goal_obj()
                attempts = 0

                # embed()
                # yumi_ar.pb_client.remove_body(obj_id)
                # start_pos = [0.4, 0.0, 0.1]
                # un_norm_ori = np.random.rand(4)
                # start_ori = un_norm_ori/(np.linalg.norm(un_norm_ori))
                # start_pose = util.list2pose_stamped(list(start_pos) + list(start_ori))
                # bandu_names = [
                #     '/root/catkin_ws/src/config/descriptions/bandu/Bandu Block/Bandu Block.urdf',
                #     '/root/catkin_ws/src/config/descriptions/bandu/Big Ring/Big Ring.urdf',
                #     '/root/catkin_ws/src/config/descriptions/bandu/Double Wedge/Double Wedge.urdf',
                #     '/root/catkin_ws/src/config/descriptions/bandu/Egg/Egg.urdf',
                #     '/root/catkin_ws/src/config/descriptions/bandu/Knight Shape/Knight Shape.urdf',
                #     '/root/catkin_ws/src/config/descriptions/bandu/Pencil/Pencil.urdf',
                #     '/root/catkin_ws/src/config/descriptions/bandu/Skewed Rectangular Prism/Skewed Rectangular Prism.urdf',
                #     '/root/catkin_ws/src/config/descriptions/bandu/Skewed Triangular Prism/Skewed Triangular Prism.urdf',
                #     '/root/catkin_ws/src/config/descriptions/bandu/Skewed Wedge/Skewed Wedge.urdf',
                # ]
                # obj_id = yumi_ar.pb_client.load_urdf(
                #     bandu_names[0],
                #     start_pos,
                #     start_ori
                # )

                # pcd1 = trimesh.PointCloud(pointcloud_pts)
                # pcd2 = trimesh.PointCloud(pointcloud_pts[np.where(start_state.pointcloud_mask)[0], :])
                # pcd1.colors = [255, 0, 0, 255]
                # pcd2.colors = [0, 0, 255, 255]
                # scene_full = trimesh.Scene([pcd1, pcd2])
                # scene1 = trimesh.Scene([pcd1])
                # scene2 = trimesh.Scene([pcd2])
                # scene_full.show()

                # embed()
                while True:
                    # if attempts > cfg.ATTEMPT_MAX:
                    if attempts > 4:
                        experiment_manager.set_mp_success(False, attempts)
                        experiment_manager.end_trial(None, False)
                        break
                    # print('attempts: ' + str(attempts))

                    attempts += 1
                    time.sleep(0.1)
                    yumi_ar.arm.go_home(ignore_physics=True)
                    if goal_visualization:
                        goal_viz.update_goal_state(util.pose_stamped2list(goal_pose))
                        goal_viz.hide_goal_obj()                    
                    time.sleep(1.0)

                    p.resetBasePositionAndOrientation(
                        obj_id,
                        util.pose_stamped2list(start_pose)[:3],
                        util.pose_stamped2list(start_pose)[3:])
                    time.sleep(1.0)

                    obs, pcd = yumi_gs.get_observation(
                        obj_id=obj_id,
                        robot_table_id=(yumi_ar.arm.robot_id, table_id))

                    goal_pose_global = util.pose_stamped2list(goal_pose)
                    goal_mat_global = util.matrix_from_pose(goal_pose)

                    start_mat = util.matrix_from_pose(start_pose)
                    T_mat_global = np.matmul(goal_mat_global, np.linalg.inv(start_mat))

                    transformation_global = util.pose_stamped2np(util.pose_from_matrix(T_mat_global))
                    # model takes in observation, and predicts:
                    pointcloud_pts = np.asarray(obs['down_pcd_pts'][:100, :], dtype=np.float32)
                    pointcloud_pts_full = np.asarray(np.concatenate(obs['pcd_pts']), dtype=np.float32)
                    table_pts_full = np.concatenate(obs['table_pcd_pts'], axis=0)

                    grasp_sampler.update_default_target(table_pts_full[::500, :])

                    # sample from model
                    start_state = PointCloudNode()
                    start_state.set_pointcloud(
                        pcd=pointcloud_pts,
                        pcd_full=pointcloud_pts_full
                    )
                    if primitive_name == 'grasp':
                        # prediction = grasp_sampler.sample(start_state.pointcloud)
                        prediction = grasp_sampler.sample(
                            state=start_state.pointcloud,
                            state_full=start_state.pointcloud_full)                        
                    elif primitive_name == 'pull':
                        prediction = pull_sampler.sample(start_state.pointcloud)
                    start_state.pointcloud_mask = prediction['mask']

                    new_state = PointCloudNode()
                    new_state.init_state(start_state, prediction['transformation'])
                    correction = False
                    if primitive_name == 'grasp':
                        correction = True
                    new_state.init_palms(prediction['palms'],
                                         correction=correction,
                                         prev_pointcloud=start_state.pointcloud_full)

                    trans_execute = util.pose_from_matrix(new_state.transformation)
                    if args.final_subgoal:
                        trans_execute = util.pose_from_matrix(T_mat_global)

                    if primitive_name == 'grasp':
                        local_plan = grasp_planning_wf(
                            util.list2pose_stamped(new_state.palms[7:]),
                            util.list2pose_stamped(new_state.palms[:7]),
                            trans_execute
                        )
                    elif primitive_name == 'pull':
                        local_plan = pulling_planning_wf(
                            util.list2pose_stamped(new_state.palms[:7]),
                            util.list2pose_stamped(new_state.palms[:7]),
                            trans_execute
                        )                        

                    if args.rviz_viz:
                        import simulation
                        for i in range(10):
                            simulation.visualize_object(
                                start_pose,
                                filepath="package://config/descriptions/meshes/objects/cuboids/" +
                                    cuboid_fname.split('objects/cuboids')[1],
                                name="/object_initial",
                                color=(1., 0., 0., 1.),
                                frame_id="/yumi_body",
                                scale=(1., 1., 1.))
                            simulation.visualize_object(
                                goal_pose,
                                filepath="package://config/descriptions/meshes/objects/cuboids/" +
                                    cuboid_fname.split('objects/cuboids')[1],
                                name="/object_final",
                                color=(0., 0., 1., 1.),
                                frame_id="/yumi_body",
                                scale=(1., 1., 1.))
                            rospy.sleep(.1)
                        simulation.simulate(local_plan, cuboid_fname.split('objects/cuboids')[1])
                        embed()
                    if args.plotly_viz:
                        plot_data = {}
                        plot_data['start'] = pointcloud_pts
                        plot_data['object_mask_down'] = start_state.pointcloud_mask

                        fig, _ = viz_pcd.plot_pointcloud(plot_data,
                                                            downsampled=True)
                        fig.show()
                        embed()
                    # embed()
                    # trans_list = []
                    # for i in range(50):
                    #     pred = pull_sampler.sample(start_state.pointcloud)
                    #     trans_list.append(util.pose_stamped2np(util.pose_from_matrix(pred['transformation'])))

                    if args.trimesh_viz:
                        viz_data = {}
                        viz_data['contact_world_frame_right'] = new_state.palms_raw[:7]
                        viz_data['contact_world_frame_left'] = new_state.palms_raw[7:]
                        # viz_data['contact_world_frame_left'] = new_state.palms_raw[:7]
                        viz_data['start_vis'] = util.pose_stamped2np(start_pose)
                        viz_data['transformation'] = util.pose_stamped2np(util.pose_from_matrix(prediction['transformation']))
                        # viz_data['transformation'] = np.asarray(trans_list).squeeze()
                        viz_data['mesh_file'] = cuboid_fname
                        viz_data['object_pointcloud'] = pointcloud_pts_full
                        viz_data['start'] = pointcloud_pts
                        # viz_data['start'] = pointcloud_pts_full
                        viz_data['object_mask'] = prediction['mask']
                        embed()

                        scene = viz_palms.vis_palms(viz_data, world=True, corr=False, full_path=True, goal_number=1)
                        scene_pcd = viz_palms.vis_palms_pcd(viz_data, world=True, corr=False, full_path=True, show_mask=True, goal_number=1)
                        scene_pcd.show()
                        # scene.show()
                        embed()

                    real_start_pos = p.getBasePositionAndOrientation(obj_id)[0]
                    real_start_ori = p.getBasePositionAndOrientation(obj_id)[1]
                    real_start_pose = list(real_start_pos) + list(real_start_ori)
                    real_start_mat = util.matrix_from_pose(util.list2pose_stamped(real_start_pose))

                    des_goal_pose = util.transform_pose(
                        util.list2pose_stamped(real_start_pose),
                        util.pose_from_matrix(prediction['transformation']))

                    if goal_visualization:
                        goal_viz.update_goal_state(util.pose_stamped2list(goal_pose))
                        goal_viz.show_goal_obj()

                    # create trial data
                    trial_data = {}
                    trial_data['start_pcd'] = pointcloud_pts_full
                    trial_data['start_pcd_down'] = pointcloud_pts
                    trial_data['start_pcd_mask'] = start_state.pointcloud_mask
                    trial_data['obj_fname'] = cuboid_fname
                    trial_data['start_pose'] = np.asarray(real_start_pose)
                    trial_data['goal_pose'] = util.pose_stamped2np(des_goal_pose)
                    trial_data['goal_pose_global'] = np.asarray(goal_pose_global)
                    trial_data['table_pcd'] = table_pts_full[::500, :]
                    trial_data['trans_des'] = util.pose_stamped2np(util.pose_from_matrix(prediction['transformation']))
                    trial_data['trans_des_global'] = transformation_global

                    # experiment_manager.start_trial()
                    action_planner.active_arm = 'right'
                    action_planner.inactive_arm = 'left'

                    if primitive_name == 'grasp':
                        # try to execute the action
                        yumi_ar.arm.set_jpos([0.9936, -2.1848, -0.9915, 0.8458, 3.7618,  1.5486,  0.1127,
                                            -1.0777, -2.1187, 0.995, 1.002, -3.6834,  1.8132,  2.6405],
                                            ignore_physics=True)
                        grasp_success = False
                        try:
                            for k, subplan in enumerate(local_plan):
                                time.sleep(1.0)
                                action_planner.playback_dual_arm('grasp', subplan, k)
                                if k > 0 and experiment_manager.still_grasping():
                                    grasp_success = True

                            real_final_pos = p.getBasePositionAndOrientation(obj_id)[0]
                            real_final_ori = p.getBasePositionAndOrientation(obj_id)[1]
                            real_final_pose = list(real_final_pos) + list(real_final_ori)
                            real_final_mat = util.matrix_from_pose(util.list2pose_stamped(real_final_pose))
                            real_T_mat = np.matmul(real_final_mat, np.linalg.inv(real_start_mat))
                            real_T_pose = util.pose_stamped2np(util.pose_from_matrix(real_T_mat))

                            trial_data['trans_executed'] = real_T_mat
                            trial_data['final_pose'] = real_final_pose
                            experiment_manager.set_mp_success(True, attempts)
                            experiment_manager.end_trial(trial_data,  grasp_success)
                            # embed()
                        except ValueError as e:
                            # print('Value Error: ', e)
                            continue
                    elif primitive_name == 'pull':
                        try:
                            yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT, ignore_physics=True)
                            time.sleep(0.5)
                            action_planner.playback_single_arm('pull', local_plan[0])
                            time.sleep(0.5)
                            action_planner.single_arm_retract()
                        except ValueError as e:
                            continue                        

                    time.sleep(3.0)
                    yumi_ar.arm.go_home(ignore_physics=True)
                    break
        embed()

        obj_data = experiment_manager.get_object_data()
        # obj_name = problems_data[problem_ind]['object_name'].split('.stl')[0]
        obj_data_fname = osp.join(
            pickle_path,
            obj_name,
            obj_name + '_eval_data.pkl')
        # print('Object data: ')
        # for key in obj_data.keys():
        #     print(key, obj_data[key])
        if args.save_data:
            print('Saving to: ' + str(obj_data_fname))
            with open(obj_data_fname, 'wb') as f:
                pickle.dump(obj_data, f)

        yumi_ar.pb_client.remove_body(obj_id)
        if goal_visualization:
            yumi_ar.pb_client.remove_body(goal_obj_id)

        # cuboid_fname = cuboid_manager.get_cuboid_fname()
        # cuboid_fname = str(osp.join(
        #     '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids',
        #     problems_data[problem_ind]['object_name']))
        while True:
            if len(prob_inds) == 0:
                print('Done with test problems!')
                return
            prob_ind = prob_inds.pop()
            obj_name = problems_data[prob_ind]['object_name'].split('.stl')[0]
            if osp.exists(osp.join(pickle_path, obj_name)):
                continue
            os.makedirs(osp.join(pickle_path, obj_name))
            break
        cuboid_fname = str(osp.join(
            '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids',
            obj_name + '.stl'))

        obj_id, sphere_ids, mesh, goal_obj_id = \
            cuboid_sampler.sample_cuboid_pybullet(
                cuboid_fname,
                goal=goal_visualization,
                keypoints=False)

        cuboid_manager.filter_collisions(obj_id, goal_obj_id)

        p.changeDynamics(
            obj_id,
            -1,
            lateralFriction=1.0
        )

        action_planner.update_object(obj_id, cuboid_fname)
        exp_single.initialize_object(obj_id, cuboid_fname)
        experiment_manager.set_object_id(
            obj_id,
            cuboid_fname
        )

        if goal_visualization:
            goal_viz.update_goal_obj(goal_obj_id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_trials',
        type=int
    )

    parser.add_argument(
        '--sim_step_repeat',
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

    parser.add_argument(
        '--num_blocks', type=int, default=20
    )

    parser.add_argument(
        '--nice_pull_release', action='store_true'
    )

    parser.add_argument(
        '--goal_viz', action='store_true'
    )

    parser.add_argument(
        '--plotly_viz', action='store_true'
    )

    parser.add_argument(
        '--rviz_viz', action='store_true'
    )

    parser.add_argument(
        '--trimesh_viz', action='store_true'
    )

    parser.add_argument(
        '--exp_name', type=str, default='debug'
    )

    parser.add_argument(
        '--pointnet', action='store_true'
    )
    
    parser.add_argument(
        '--final_subgoal', action='store_true'
    )

    args = parser.parse_args()
    main(args)
