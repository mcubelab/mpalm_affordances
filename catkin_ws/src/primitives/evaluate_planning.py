import os
import os.path as osp
import sys
import time
import argparse
import numpy as np
import rospy
import signal
import threading
import pickle
import open3d
import copy
from IPython import embed

from airobot import Robot
from airobot.utils import common
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions
from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

# from closed_loop_experiments_cfg import get_cfg_defaults
from multistep_planning_eval_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler
from data_gen_utils import YumiCamsGS, DataManager, MultiBlockManager, GoalVisual
import simulation
from helper import registration as reg
from helper.pointcloud_planning import (
    PointCloudNode, PointCloudTree,
    GraspSamplerVAEPubSub, PullSamplerVAEPubSub,
    PullSamplerBasic, GraspSamplerBasic,
    GraspSkill, PullRightSkill)
from planning import grasp_planning_wf, pulling_planning_wf
from eval_utils.visualization_tools import PCDVis, PalmVis
from eval_utils.experiment_recorder import GraspEvalManager


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


def main(args):
    cfg_file = osp.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    rospy.init_node('EvalMultiStep')
    signal.signal(signal.SIGINT, signal_handler)

    data_seed = args.np_seed
    primitive_name = args.primitive

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

    yumi_gs = YumiCamsGS(
        yumi_ar,
        cfg,
        exec_thread=False,
        sim_step_repeat=args.sim_step_repeat
    )

    for _ in range(10):
        yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    cuboid_sampler = CuboidSampler(
        osp.join(
            os.environ['CODE_BASE'],
            'catkin_ws/src/config/descriptions/meshes/objects/cuboids/nominal_cuboid.stl'),
        pb_client=yumi_ar.pb_client)
    cuboid_fname_template = osp.join(
        os.environ['CODE_BASE'],
        'catkin_ws/src/config/descriptions/meshes/objects/cuboids/')

    cuboid_manager = MultiBlockManager(
        cuboid_fname_template,
        cuboid_sampler,
        robot_id=yumi_ar.arm.robot_id,
        table_id=table_id,
        r_gel_id=r_gel_id,
        l_gel_id=l_gel_id)

    if args.multi:
        cuboid_fname = cuboid_manager.get_cuboid_fname()
        # cuboid_fname = '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids/test_cuboid_smaller_4479.stl'
    else:
        cuboid_fname = args.config_package_path + 'descriptions/meshes/objects/' + \
            args.object_name + '_experiments.stl'
    mesh_file = cuboid_fname

    goal_visualization = False
    if args.goal_viz:
        goal_visualization = True

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

    # goal_face = 0
    goal_faces = [0, 1, 2, 3, 4, 5]
    from random import shuffle
    shuffle(goal_faces)
    goal_face = goal_faces[0]

    # exp_single = SingleArmPrimitives(
    #     cfg,
    #     yumi_ar.pb_client.get_client_id(),
    #     obj_id,
    #     cuboid_fname)
    # k = 0
    # while True:
    #     k += 1
    #     if k > 10:
    #         print('FAILED TO BUILD GRASPING GRAPH')
    #         return
    #     try:
    #         exp_double = DualArmPrimitives(
    #             cfg,
    #             yumi_ar.pb_client.get_client_id(),
    #             obj_id,
    #             cuboid_fname,
    #             goal_face=goal_face)
    #         break
    #     except ValueError as e:
    #         print(e)
    #         yumi_ar.pb_client.remove_body(obj_id)
    #         if goal_visualization:
    #             yumi_ar.pb_client.remove_body(goal_obj_id)
    #         cuboid_fname = cuboid_manager.get_cuboid_fname()
    #         print("Cuboid file: " + cuboid_fname)

    #         obj_id, sphere_ids, mesh, goal_obj_id = \
    #             cuboid_sampler.sample_cuboid_pybullet(
    #                 cuboid_fname,
    #                 goal=goal_visualization,
    #                 keypoints=False)

    #         cuboid_manager.filter_collisions(obj_id, goal_obj_id)

    #         p.changeDynamics(
    #             obj_id,
    #             -1,
    #             lateralFriction=1.0)

    # if primitive_name == 'grasp':
    #     exp_running = exp_double
    # else:
    #     exp_running = exp_single

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
            cfg.OBJECT_POSE_3,
            show_init=False)

    action_planner.update_object(obj_id, mesh_file)
    # exp_single.initialize_object(obj_id, cuboid_fname)

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
    pred_dir = osp.join(os.environ['CODE_BASE'], cfg.PREDICTION_DIR)
    obs_dir = osp.join(os.environ['CODE_BASE'], cfg.OBSERVATION_DIR)

    if args.save_data:
        with open(osp.join(pickle_path, 'metadata.pkl'), 'wb') as mdata_f:
            pickle.dump(metadata, mdata_f)

    total_trials = 0
    successes = 0

    # prep visualization tools
    palm_mesh_file = osp.join(os.environ['CODE_BASE'],
                              cfg.PALM_MESH_FILE)
    table_mesh_file = osp.join(os.environ['CODE_BASE'],
                               cfg.TABLE_MESH_FILE)
    viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
    viz_pcd = PCDVis()

    if args.skeleton == 'pg':
        skeleton = ['pull', 'grasp']
    elif args.skeleton == 'gp':
        skeleton = ['grasp', 'pull']
    elif args.skeleton == 'pgp':
        skeleton = ['pull', 'grasp', 'pull'] 
    else:
        raise ValueError('Unrecognized plan skeleton!')

    pull_sampler = PullSamplerBasic()
    # grasp_sampler = GraspSamplerBasic(None)
    # pull_sampler = PullSamplerVAEPubSub(
    #     obs_dir=obs_dir,
    #     pred_dir=pred_dir
    # )

    yumi_ar.pb_client.remove_body(obj_id)
    if goal_visualization:
        yumi_ar.pb_client.remove_body(goal_obj_id)

    # if args.bookshelf:
    obs, pcd = yumi_gs.get_observation(
        obj_id=obj_id,
        robot_table_id=(yumi_ar.arm.robot_id, 28))
    shelf_pcd = open3d.geometry.PointCloud()
    shelf_pcd.points = open3d.utility.Vector3dVector(np.concatenate(obs['table_pcd_pts']))
    shelf_pointcloud = np.asarray(shelf_pcd.points)
    z_sort = np.sort(shelf_pointcloud[:, 2])[::-1]
    top_z_2 = z_sort[10]
    target_surface = shelf_pointcloud[np.where(shelf_pointcloud[:, 2] > 0.9*top_z_2)[0], :]

    grasp_sampler = GraspSamplerVAEPubSub(
        default_target=target_surface,
        obs_dir=obs_dir,
        pred_dir=pred_dir
    )

    pull_skill = PullRightSkill(pull_sampler, yumi_gs, pulling_planning_wf)
    pull_skill_no_mp = PullRightSkill(pull_sampler, yumi_gs,
                                      pulling_planning_wf, ignore_mp=True)
    grasp_skill = GraspSkill(grasp_sampler, yumi_gs, grasp_planning_wf)
    skills = {}
    # skills['pull'] = pull_skill_no_mp
    skills['pull'] = pull_skill
    skills['grasp'] = grasp_skill

    # problems_file = '/root/catkin_ws/src/primitives/data/planning/test_problems_0/demo_0.pkl'
    problems_file = '/root/catkin_ws/src/primitives/data/planning/bookshelf_2/bookshelf_problems.pkl'
    with open(problems_file, 'rb') as f:
        problems_data = pickle.load(f)

    prob_inds = np.arange(len(problems_data), dtype=np.int64).tolist()
    data_inds = np.arange(len(problems_data[0]['problems']), dtype=np.int64).tolist()

    experiment_manager = GraspEvalManager(
        yumi_gs,
        yumi_ar.pb_client.get_client_id(),
        pickle_path,
        args.exp_name,
        None,
        None,
        None,
        None,
        cfg
    )

    # experiment_manager.set_object_id(
    #     obj_id,
    #     cuboid_fname
    # )

    total_trial_number = 0
    for _ in range(len(problems_data)):
        # prob_ind = 3

        # obj_fname = str(osp.join(
        #     '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids',
        #     problems_data[prob_ind]['object_name']))

        # print(obj_fname)
        # for j, problem_data in enumerate(problems_data[prob_ind]['problems']):
        for _ in range(len(problems_data[0]['problems'])):
            total_trial_number += 1
            # prob_ind = 8
            # data_ind = 15
            prob_ind = prob_inds[np.random.randint(len(prob_inds))]
            data_ind = data_inds[np.random.randint(len(data_inds))]
            # problem_data = problems_data[prob_ind]['problems'][data_ind]
            # problem_data = problems_data[prob_ind]['problems']

            problem_data = problems_data[prob_ind]
            embed()
            # start_pose = problem_data['problems']['start_vis']
            # goal_pose = problem_data['problems']['goal_vis']
            # transformation = problem_data['problems']['transformation']
            # stl_file = problem_data['object_name']
            # book_scale = problem_data['object_scale']

            # obj_fname = str(osp.join(
            #     '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids',
            #     problems_data[prob_ind]['object_name']))
            # obj_name = problems_data[prob_ind]['object_name'].split('.stl')[0]
            # book_scale = None

            stl_file = problem_data['object_name']
            obj_fname = stl_file
            obj_name = obj_fname.split('.stl')[0].split('/meshes/objects/')[1]
            book_scale = problem_data['object_scale']

            # yumi_ar.pb_client.remove_body(obj_id)
            # yumi_ar.pb_client.remove_body(goal_obj_id)

            print(obj_fname, obj_name)
            start_pose = problem_data['problems']['start_vis'].tolist()
            goal_pose = problem_data['problems']['goal_vis'].tolist()

            # put object into work at start_pose, with known obj_fname
            yumi_ar.pb_client.remove_body(obj_id)
            if goal_visualization:
                yumi_ar.pb_client.remove_body(goal_obj_id)

            obj_id, sphere_ids, mesh, goal_obj_id = \
                cuboid_sampler.sample_cuboid_pybullet(
                    obj_fname,
                    goal=goal_visualization,
                    keypoints=False,
                    scale=book_scale)

            if goal_visualization:
                goal_viz.update_goal_obj(goal_obj_id)
                goal_viz.update_goal_state(goal_pose)
                # goal_viz.hide_goal_obj()
                cuboid_manager.filter_collisions(obj_id, goal_obj_id)

            # exp_single.initialize_object(obj_id, obj_fname)
            experiment_manager.set_object_id(
                obj_id,
                obj_fname
            )

            p.resetBasePositionAndOrientation(
                obj_id,
                start_pose[:3],
                start_pose[3:])

            p.changeDynamics(
                obj_id,
                -1,
                lateralFriction=1.0
            )

            yumi_ar.arm.set_jpos(cfg.RIGHT_OUT_OF_FRAME +
                                 cfg.LEFT_OUT_OF_FRAME,
                                 ignore_physics=True)
            time.sleep(0.2)

            real_start_pos = p.getBasePositionAndOrientation(obj_id)[0]
            real_start_ori = p.getBasePositionAndOrientation(obj_id)[1]
            real_start_pose = list(real_start_pos) + list(real_start_ori)

            # transformation_des = util.matrix_from_pose(
            #     util.list2pose_stamped(problem_data['transformation'].tolist()))
            transformation_des = problem_data['problems']['transformation']

            # #### BEGIN BLOCK FOR GETTING INTRO FIGURE
            # R_3 = common.euler2rot([0.0, 0.0, np.pi/4])
            # t_3 = np.array([0.03, 0.25, 0.0])
            # T_3 = np.eye(4)
            # T_3[:-1, :-1] = R_3
            # # T_3[:-1, -1] = t_3
            # print(T_3)
            # trans_des = np.matmul(T_3, transformation_des)


            # goal_pose = util.pose_stamped2list(util.transform_pose(
            #     util.list2pose_stamped(real_start_pose),
            #     util.pose_from_matrix(trans_des)
            # ))
            # if goal_visualization:
            #     goal_viz.update_goal_state(goal_pose)
            #     goal_viz.show_goal_obj()

            # embed()
            # #### END BLOCK FOR GETTING INTRO FIGURE

            # goal_pose = util.pose_stamped2list(util.transform_pose(
            #     util.list2pose_stamped(real_start_pose),
            #     util.list2pose_stamped(problem_data['transformation'])
            # ))

            # if skeleton is 'pull' 'grasp' 'pull', add an additional SE(2) transformation to the task
            if args.skeleton == 'pgp':
                # while True:
                #     x, y, dq = exp_single.get_rand_trans_yaw()

                #     goal_pose_2_list = copy.deepcopy(goal_pose)
                #     goal_pose_2_list[0] = x
                #     goal_pose_2_list[1] = y
                #     goal_pose_2_list[3:] = common.quat_multiply(dq, np.asarray(goal_pose[3:]))

                #     if goal_pose_2_list[0] > 0.2 and goal_pose_2_list[0] < 0.4 and \
                #             goal_pose_2_list[1] > -0.3 and goal_pose_2_list[1] < 0.1:
                #         rot = common.quat2rot(dq)
                #         T_2 = np.eye(4)
                #         T_2[:-1, :-1] = rot
                #         T_2[:2, -1] = [x-goal_pose[0], y-goal_pose[1]]
                #         break

                # goal_pose = goal_pose_2_list
                # transformation_des = np.matmul(T_2, transformation_des)
                pass

            # # if skeleton is 'grasp' first, invert the desired trans and flip everything
            if args.skeleton == 'gp':
                transformation_des = np.linalg.inv(transformation_des)
                start_tmp = copy.deepcopy(start_pose)
                start_pose = goal_pose
                goal_pose = start_tmp

                p.resetBasePositionAndOrientation(
                    obj_id,
                    start_pose[:3],
                    start_pose[3:])

                real_start_pos = p.getBasePositionAndOrientation(obj_id)[0]
                real_start_ori = p.getBasePositionAndOrientation(obj_id)[1]
                real_start_pose = list(real_start_pos) + list(real_start_ori)

                time.sleep(0.5)

            # get observation
            obs, pcd = yumi_gs.get_observation(
                obj_id=obj_id,
                robot_table_id=(yumi_ar.arm.robot_id, table_id))

            pointcloud_pts = np.asarray(obs['down_pcd_pts'][:100, :], dtype=np.float32)
            pointcloud_pts_full = np.asarray(np.concatenate(obs['pcd_pts']), dtype=np.float32)

            # grasp_sampler.update_default_target(
            #     np.concatenate(obs['table_pcd_pts'], axis=0)[::500, :])


            trial_data = {}
            trial_data['start_pcd'] = pointcloud_pts_full
            trial_data['start_pcd_down'] = pointcloud_pts
            trial_data['obj_fname'] = cuboid_fname
            trial_data['start_pose'] = np.asarray(real_start_pose)
            trial_data['goal_pose'] = np.asarray(goal_pose)
            trial_data['goal_pose_global'] = np.asarray(goal_pose)
            trial_data['trans_des_global'] = transformation_des

            trial_data['skeleton'] = args.skeleton

            # plan!
            planner = PointCloudTree(
                pointcloud_pts,
                transformation_des,
                skeleton,
                skills,
                start_pcd_full=pointcloud_pts_full,
                visualize=True,
                obj_id=goal_obj_id,
                start_pose=util.list2pose_stamped(start_pose))
            start_plan_time = time.time()
            plan_total = planner.plan()
            trial_data['planning_time'] = time.time() - start_plan_time

            if plan_total is None:
                print('Could not find plan')
                experiment_manager.set_mp_success(False, 1)
                obj_data = experiment_manager.get_object_data()
                # obj_data_fname = osp.join(
                #     pickle_path,
                #     obj_name + '_' + str(total_trial_number),
                #     obj_name + '_' + str(total_trial_number) + '_ms_eval_data.pkl')
                obj_data_fname = osp.join(
                    pickle_path,
                    obj_name + '_' + str(total_trial_number) + '_ms_eval_data.pkl')
                if args.save_data:
                    print('Saving to: ' + str(obj_data_fname))
                    with open(obj_data_fname, 'wb') as f:
                        pickle.dump(obj_data, f)
                continue

            plan = copy.deepcopy(plan_total[1:])

            if args.trimesh_viz:
                # from multistep_planning_eval_cfg import get_cfg_defaults
                # import os.path as osp
                # from eval_utils.visualization_tools import PCDVis, PalmVis
                # cfg = get_cfg_defaults()
                # palm_mesh_file = osp.join(os.environ['CODE_BASE'],
                #                         cfg.PALM_MESH_FILE)
                # table_mesh_file = osp.join(os.environ['CODE_BASE'],
                #                         cfg.TABLE_MESH_FILE)
                # viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
                # viz_pcd = PCDVis()

                ind = 0

                pcd_data = copy.deepcopy(problem_data)
                pcd_data['start'] = plan_total[ind].pointcloud_full
                pcd_data['object_pointcloud'] = plan_total[ind].pointcloud_full
                pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(plan_total[ind+1].transformation)))
                pcd_data['contact_world_frame_right'] = np.asarray(plan_total[ind+1].palms[:7])
                if skeleton[ind] == 'grasp':
                    pcd_data['contact_world_frame_left'] = np.asarray(plan_total[ind+1].palms[:7])
                elif skeleton[ind] == 'pull':
                    pcd_data['contact_world_frame_left'] = np.asarray(plan_total[ind+1].palms[7:])
                scene = viz_palms.vis_palms_pcd(pcd_data, world=True, centered=False, corr=False)
                scene.show()

                # embed()

            # execute plan if one is found...
            pose_plan = [(real_start_pose, util.list2pose_stamped(real_start_pose))]
            for i in range(1, len(plan)+1):
                pose = util.transform_pose(pose_plan[i-1][1], util.pose_from_matrix(plan[i-1].transformation))
                pose_list = util.pose_stamped2list(pose)
                pose_plan.append((pose_list, pose))


            # get palm poses from plan
            palm_pose_plan = []
            for i, node in enumerate(plan):
                palms = copy.deepcopy(node.palms)
                # palms = copy.deepcopy(node.palms_raw) if node.palms_raw is not None else copy.deepcopy(node.palms)
                if skeleton[i] == 'pull':
                    palms[2] -= 0.002
                palm_pose_plan.append(palms)

            # observe results
            full_plan = []
            for i in range(len(plan)):
                if skeleton[i] == 'pull':
                    local_plan = pulling_planning_wf(
                        util.list2pose_stamped(palm_pose_plan[i]),
                        util.list2pose_stamped(palm_pose_plan[i]),
                        util.pose_from_matrix(plan[i].transformation)
                    )
                elif skeleton[i] == 'grasp':
                    local_plan = grasp_planning_wf(
                        util.list2pose_stamped(palm_pose_plan[i][7:]),
                        util.list2pose_stamped(palm_pose_plan[i][:7]),
                        util.pose_from_matrix(plan[i].transformation)
                    )
                full_plan.append(local_plan)


            grasp_success = True

            action_planner.active_arm = 'right'
            action_planner.inactive_arm = 'left'

            if goal_visualization:
                goal_viz.update_goal_state(goal_pose)
                goal_viz.show_goal_obj()

            if goal_visualization:
                goal_viz.update_goal_state(goal_pose)
                goal_viz.show_goal_obj()

            real_start_pos = p.getBasePositionAndOrientation(obj_id)[0]
            real_start_ori = p.getBasePositionAndOrientation(obj_id)[1]
            real_start_pose = list(real_start_pos) + list(real_start_ori)
            real_start_mat = util.matrix_from_pose(util.list2pose_stamped(real_start_pose))

            # embed()
            try:
                start_playback_time = time.time()
                for playback in range(args.playback_num):
                    if playback > 0 and goal_visualization:
                        goal_viz.hide_goal_obj()

                    yumi_ar.pb_client.reset_body(obj_id, pose_plan[0][0][:3], pose_plan[0][0][3:])
                    p.changeDynamics(
                        obj_id,
                        -1,
                        lateralFriction=1.0
                    )
                    for i, skill in enumerate(skeleton):
                        if skill == 'pull':
                            # set arm configuration to good start state
                            yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT, ignore_physics=True)
                            time.sleep(0.5)

                            # move to making contact, and ensure contact is made
                            # try:
                            #     _, _ = action_planner.single_arm_setup(full_plan[i][0], pre=True)
                            # except ValueError as e:
                            #     print(e)
                            #     break
                            _, _ = action_planner.single_arm_setup(full_plan[i][0], pre=True)
                            start_playback_time = time.time()
                            if not experiment_manager.still_pulling():
                                while True:
                                    if experiment_manager.still_pulling() or time.time() - start_playback_time > 20.0:
                                        break
                                    action_planner.single_arm_approach()
                                    time.sleep(0.075)
                                    new_plan = pulling_planning_wf(
                                        yumi_gs.get_current_tip_poses()['left'],
                                        yumi_gs.get_current_tip_poses()['right'],
                                        util.pose_from_matrix(plan[i].transformation)
                                    )
                                pull_plan = new_plan[0]
                            else:
                                pull_plan = full_plan[i][0]
                            # try:
                            #     action_planner.playback_single_arm('pull', pull_plan, pre=False)
                            # except ValueError as e:
                            #     print(e)
                            #     break
                            action_planner.playback_single_arm('pull', pull_plan, pre=False)
                            grasp_success = grasp_success and experiment_manager.still_pulling(n=False)
                            print('grasp success: ' + str(grasp_success))
                            time.sleep(0.5)
                            action_planner.single_arm_retract()

                        elif skill == 'grasp':
                            yumi_ar.arm.set_jpos([0.9936, -2.1848, -0.9915, 0.8458, 3.7618,  1.5486,  0.1127,
                                                -1.0777, -2.1187, 0.995, 1.002,  -3.6834,  1.8132,  2.6405],
                                                ignore_physics=True)
                            time.sleep(0.5)
                            # try:
                            #     _, _ = action_planner.dual_arm_setup(full_plan[i][0], 0, pre=True)
                            # except ValueError as e:
                            #     print(e)
                            #     break
                            _, _ = action_planner.dual_arm_setup(full_plan[i][0], 0, pre=True)
                            start_playback_time = time.time()
                            if not experiment_manager.still_grasping():
                                while True:
                                    if experiment_manager.still_grasping() or time.time() - start_playback_time > 20.0:
                                        break
                                    action_planner.dual_arm_approach()
                                    time.sleep(0.075)
                                    new_plan = grasp_planning_wf(
                                        yumi_gs.get_current_tip_poses()['left'],
                                        yumi_gs.get_current_tip_poses()['right'],
                                        util.pose_from_matrix(plan[i].transformation)
                                    )
                                grasp_plan = new_plan
                            else:
                                grasp_plan = full_plan[i]
                            for k, subplan in enumerate(grasp_plan):
                                # try:
                                #     action_planner.playback_dual_arm('grasp', subplan, k, pre=False)
                                # except ValueError as e:
                                #     print(e)
                                #     break
                                action_planner.playback_dual_arm('grasp', subplan, k, pre=False)
                                if k == 1:
                                    grasp_success = grasp_success and experiment_manager.still_grasping(n=False)
                                    print('grasp success: ' + str(grasp_success))
                                time.sleep(1.0)
            except ValueError as e:
                print(e)
                experiment_manager.set_mp_success(True, 1)
                experiment_manager.set_execute_success(False)
                obj_data = experiment_manager.get_object_data()
                # obj_data_fname = osp.join(
                #     pickle_path,
                #     obj_name + '_' + str(total_trial_number),
                #     obj_name + '_' + str(total_trial_number) + '_ms_eval_data.pkl')
                obj_data_fname = osp.join(
                    pickle_path,
                    obj_name + '_' + str(total_trial_number) + '_ms_eval_data.pkl')
                if args.save_data:
                    print('Saving to: ' + str(obj_data_fname))
                    with open(obj_data_fname, 'wb') as f:
                        pickle.dump(obj_data, f)
                continue


            real_final_pos = p.getBasePositionAndOrientation(obj_id)[0]
            real_final_ori = p.getBasePositionAndOrientation(obj_id)[1]
            real_final_pose = list(real_final_pos) + list(real_final_ori)
            real_final_mat = util.matrix_from_pose(util.list2pose_stamped(real_final_pose))
            real_T_mat = np.matmul(real_final_mat, np.linalg.inv(real_start_mat))
            real_T_pose = util.pose_stamped2np(util.pose_from_matrix(real_T_mat))

            trial_data['trans_executed'] = real_T_mat
            trial_data['final_pose'] = real_final_pose

            experiment_manager.set_mp_success(True, 1)
            experiment_manager.set_execute_success(True)
            experiment_manager.end_trial(trial_data, grasp_success)

            time.sleep(3.0)

            obj_data = experiment_manager.get_object_data()

            kvs = {}
            kvs['grasp_success'] = obj_data['grasp_success']
            kvs['pos_err'] = np.mean(obj_data['final_pos_error'])
            kvs['ori_err'] = np.mean(obj_data['final_ori_error'])
            kvs['planning_time'] = obj_data['planning_time']
            string = ''

            for k, v in kvs.items():
                string += "%s: %.3f, " % (k,v)
            print(string)

            # obj_data_fname = osp.join(
            #     pickle_path,
            #     obj_name + '_' + str(total_trial_number),
            #     obj_name + '_' + str(total_trial_number)  + '_ms_eval_data.pkl')
            obj_data_fname = osp.join(
                pickle_path,
                obj_name + '_' + str(total_trial_number)  + '_ms_eval_data.pkl')
            if args.save_data:
                print('Saving to: ' + str(obj_data_fname))
                with open(obj_data_fname, 'wb') as f:
                    pickle.dump(obj_data, f)

            # embed()


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
        '--trimesh_viz', action='store_true'
    )

    parser.add_argument(
        '--skeleton', type=str, default='pg'
    )

    parser.add_argument(
        '--playback_num', type=int, default=1
    )

    parser.add_argument(
        '--exp_name', type=str, default='debug'
    )

    args = parser.parse_args()
    main(args)
