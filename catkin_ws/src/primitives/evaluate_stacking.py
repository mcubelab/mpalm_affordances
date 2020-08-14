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
from helper.pointcloud_planning import PointCloudTree
from helper.pointcloud_planning_utils import PointCloudNode
from helper.pull_samplers import PullSamplerBasic, PullSamplerVAEPubSub
from helper.grasp_samplers import GraspSamplerVAEPubSub, GraspSamplerBasic
from helper.push_samplers import PushSamplerVAEPubSub
from helper.skills import GraspSkill, PullRightSkill, PullLeftSkill, PushRightSkill

from planning import grasp_planning_wf, pulling_planning_wf, pushing_planning_wf
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

    # skeleton = ['pull_right', 'pull_left']
    # skeleton = ['pull_right', 'grasp', 'pull_right', 'grasp_pp', 'pull_left']
    skeleton = ['pull_right', 'grasp', 'pull_left']

    # pull_sampler = PullSamplerBasic()
    # grasp_sampler = GraspSamplerBasic(None)
    pull_sampler = PullSamplerVAEPubSub(
        obs_dir=obs_dir,
        pred_dir=pred_dir
    )

    push_sampler = PushSamplerVAEPubSub(
        obs_dir=obs_dir,
        pred_dir=pred_dir
    )

    yumi_ar.pb_client.remove_body(obj_id)
    if goal_visualization:
        yumi_ar.pb_client.remove_body(goal_obj_id)

    # if args.bookshelf:
    # obs, pcd = yumi_gs.get_observation(
    #     obj_id=obj_id,
    #     robot_table_id=(yumi_ar.arm.robot_id, 28))
    # shelf_pcd = open3d.geometry.PointCloud()
    # shelf_pcd.points = open3d.utility.Vector3dVector(np.concatenate(obs['table_pcd_pts']))
    # shelf_pointcloud = np.asarray(shelf_pcd.points)
    # z_sort = np.sort(shelf_pointcloud[:, 2])[::-1]
    # top_z_2 = z_sort[10]
    # shelf_target_surface = shelf_pointcloud[np.where(shelf_pointcloud[:, 2] > 0.9*top_z_2)[0], :]
    # target_surface_skeleton = [None, None, None, shelf_target_surface, shelf_target_surface]
    
    target_surface_skeleton = None

    # grasp_sampler = GraspSamplerVAEPubSub(
    #     default_target=target_surface,
    #     obs_dir=obs_dir,
    #     pred_dir=pred_dir
    # )
    grasp_sampler = GraspSamplerVAEPubSub(
        default_target=None,
        obs_dir=obs_dir,
        pred_dir=pred_dir
    )

    # pull_skill = PullRightSkill(pull_sampler, yumi_gs, pulling_planning_wf)
    pull_right_skill = PullRightSkill(
        pull_sampler,
        yumi_gs,
        pulling_planning_wf,
        ignore_mp=False,
        avoid_collisions=True
    )

    pull_left_skill = PullLeftSkill(
        pull_sampler,
        yumi_gs,
        pulling_planning_wf,
        ignore_mp=False,
        avoid_collisions=True
    )

    push_right_skill = PushRightSkill(
        push_sampler,
        yumi_gs,
        pushing_planning_wf,
        ignore_mp=False,
        avoid_collisions=True
    )
    # pull_skill_no_mp = PullRightSkill(pull_sampler, yumi_gs,
    #                                   pulling_planning_wf, ignore_mp=True)
    grasp_skill = GraspSkill(grasp_sampler, yumi_gs, grasp_planning_wf)
    grasp_pp_skill = GraspSkill(grasp_sampler, yumi_gs, grasp_planning_wf, pp=True)
    skills = {}
    # skills['pull'] = pull_skill_no_mp
    # skills['pull'] = pull_skill
    skills['pull_right'] = pull_right_skill
    skills['pull_left'] = pull_left_skill
    skills['grasp'] = grasp_skill
    skills['grasp_pp'] = grasp_pp_skill
    skills['push_right'] = push_right_skill


    # problems_file = '/root/catkin_ws/src/primitives/data/planning/test_problems_0/demo_1.pkl'
    # problems_file = '/root/catkin_ws/src/primitives/data/planning/bookshelf_1/bookshelf_problems.pkl'
    # problems_file = '/root/catkin_ws/src/primitives/data/planning/bookshelf_cuboid/bookshelf_problems.pkl'    
    # problems_file = '/root/catkin_ws/src/primitives/data/planning/gen_obj_1/gen_obj_problems_0.pkl'
    problems_file = '/root/catkin_ws/src/primitives/data/planning/stacking_cuboids_0/stacking_cuboids_problems_0.pkl'
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
        for _ in range(len(problems_data[0]['problems'])):
            total_trial_number += 1

            prob_ind = prob_inds[np.random.randint(len(prob_inds))]
            data_ind = data_inds[np.random.randint(len(data_inds))]

            problem_data = problems_data[prob_ind]
            stl_file = problem_data['object_name']
            obj_fname = stl_file
            obj_name = obj_fname.split('.stl')[0].split('/meshes/objects/')[1]
            scale = problem_data['object_scale']
            start_pose = problem_data['problems']['start_vis'].tolist()
            goal_pose = problem_data['problems']['goal_vis'].tolist()
            transformation_des = problem_data['problems']['transformation']

            stl_file2 = problem_data['object_name2']
            obj_fname2 = stl_file2
            obj_name2 = obj_fname2.split('.stl')[0].split('/meshes/objects/')[1]
            start_pose2 = problem_data['problems']['start_vis2'].tolist()
            goal_pose2 = problem_data['problems']['goal_vis2'].tolist()
            transformation_des2 = problem_data['problems']['transformation2']            

            # put object into work at start_pose, with known obj_fname
            yumi_ar.pb_client.remove_body(obj_id)
            if goal_visualization:
                yumi_ar.pb_client.remove_body(goal_obj_id)

            obj_id, sphere_ids, mesh, goal_obj_id = \
                cuboid_sampler.sample_cuboid_pybullet(
                    obj_fname,
                    goal=goal_visualization,
                    keypoints=False,
                    scale=scale)

            obj_id2, sphere_ids2, mesh2, goal_obj_id2 = \
                cuboid_sampler.sample_cuboid_pybullet(
                    obj_fname2,
                    goal=goal_visualization,
                    keypoints=False,
                    scale=scale)                    

            if goal_visualization:
                goal_viz.update_goal_obj(goal_obj_id)
                goal_viz.update_goal_state(goal_pose)
                # goal_viz.hide_goal_obj()
                cuboid_manager.filter_collisions(obj_id, goal_obj_id)
                cuboid_manager.filter_collisions(obj_id2, goal_obj_id2)
                p.setCollisionFilterPair(goal_obj_id2, goal_obj_id, -1, -1, enableCollision=True) 
                p.setCollisionFilterPair(obj_id2, obj_id, -1, -1, enableCollision=True)                                

            # embed()

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


            p.resetBasePositionAndOrientation(
                obj_id2,
                start_pose2[:3],
                start_pose2[3:])

            p.changeDynamics(
                obj_id2,
                -1,
                lateralFriction=1.0
            )  

            p.resetBasePositionAndOrientation(
                goal_obj_id,
                goal_pose[:3],
                goal_pose[3:])

            p.resetBasePositionAndOrientation(
                goal_obj_id2,
                goal_pose2[:3],
                goal_pose2[3:])
              

            yumi_ar.arm.set_jpos(cfg.RIGHT_OUT_OF_FRAME +
                                 cfg.LEFT_OUT_OF_FRAME,
                                 ignore_physics=True)
            time.sleep(0.5)

            real_start_pos = p.getBasePositionAndOrientation(obj_id)[0]
            real_start_ori = p.getBasePositionAndOrientation(obj_id)[1]
            real_start_pose = list(real_start_pos) + list(real_start_ori)

            real_start_pos2 = p.getBasePositionAndOrientation(obj_id2)[0]
            real_start_ori2 = p.getBasePositionAndOrientation(obj_id2)[1]
            real_start_pose2 = list(real_start_pos2) + list(real_start_ori2)            

            real_goal_pos = p.getBasePositionAndOrientation(goal_obj_id)[0]
            real_goal_ori = p.getBasePositionAndOrientation(goal_obj_id)[1]
            real_goal_pose = list(real_goal_pos) + list(real_goal_ori)

            real_goal_pos2 = p.getBasePositionAndOrientation(goal_obj_id2)[0]
            real_goal_ori2 = p.getBasePositionAndOrientation(goal_obj_id2)[1]
            real_goal_pose2 = list(real_goal_pos2) + list(real_goal_ori2)            

            transformation_des = np.matmul(
                util.matrix_from_pose(util.list2pose_stamped(real_goal_pose)), 
                np.linalg.inv(util.matrix_from_pose(util.list2pose_stamped(real_start_pose))))  

            transformation_des2 = np.matmul(
                util.matrix_from_pose(util.list2pose_stamped(real_goal_pose2)), 
                np.linalg.inv(util.matrix_from_pose(util.list2pose_stamped(real_start_pose2))))                            
          
            embed()

            # get observation
            obs, pcd = yumi_gs.get_observation(
                obj_id=obj_id,
                robot_table_id=(yumi_ar.arm.robot_id, table_id))             

            pointcloud_pts = np.asarray(obs['down_pcd_pts'][:100, :], dtype=np.float32)
            pointcloud_pts_full = np.asarray(np.concatenate(obs['pcd_pts']), dtype=np.float32)

            # get observation
            obs2, pcd = yumi_gs.get_observation(
                obj_id=obj_id2,
                robot_table_id=(yumi_ar.arm.robot_id, table_id))   

            pointcloud_pts2 = np.asarray(obs2['down_pcd_pts'][:100, :], dtype=np.float32)
            pointcloud_pts_full2 = np.asarray(np.concatenate(obs2['pcd_pts']), dtype=np.float32)

            grasp_sampler.update_default_target(
                np.concatenate(obs['table_pcd_pts'], axis=0)[::500, :])

            # plan!
            # skeleton = ['pull_right', 'grasp', 'pull_right']
            skeleton = ['pull_left', 'grasp', 'pull_left']
            planner = PointCloudTree(
                pointcloud_pts,
                transformation_des,
                skeleton,
                skills,
                start_pcd_full=pointcloud_pts_full,
                visualize=True,
                obj_id=goal_obj_id,
                start_pose=util.list2pose_stamped(start_pose),
                target_surfaces=target_surface_skeleton)
            plan_total = planner.plan()
            embed()

            base_pcd = plan_total[-1].pointcloud_full
            z_sort = np.sort(base_pcd[:, 2])[::-1]
            top_z_2 = z_sort[10]            
            base_pcd_surface = base_pcd[np.where(base_pcd[:, 2] > 0.9*top_z_2)[0], :]            

            target_surface_skeleton2 = [None, None, None, base_pcd_surface, None]

            skeleton2 = ['pull_right', 'grasp', 'pull_right', 'grasp_pp', 'pull_left']
            planner2 = PointCloudTree(
                pointcloud_pts2,
                transformation_des2,
                skeleton2,
                skills,
                start_pcd_full=pointcloud_pts_full2,
                visualize=True,
                obj_id=goal_obj_id2,
                start_pose=util.list2pose_stamped(start_pose2),
                target_surfaces=target_surface_skeleton2)
            plan_total2 = planner2.plan()

            embed()
            plan_total_list = [plan_total, plan_total2]
            skeleton_list = [skeleton, skeleton2]
            obj_id_list = [obj_id, obj_id2]
            goal_obj_id_list = [goal_obj_id, goal_obj_id2]
            obj_fname_list = [obj_fname, obj_fname2]
            mesh_list = [mesh, mesh2]
            start_pose_list = [real_start_pose, real_start_pose2]

            for plan_ind in range(len(plan_total_list)):

                plan_total_list = [plan_total, plan_total2]
                # skeleton_list = [skeleton, skeleton2]
                # obj_id_list = [obj_id, obj_id2]
                # goal_obj_id_list = [goal_obj_id, goal_obj_id2]

                plan = copy.deepcopy(plan_total_list[plan_ind][1:])
                skeleton = skeleton_list[plan_ind]
                obj_id = obj_id_list[plan_ind]
                goal_obj_id = goal_obj_id_list[plan_ind]
                obj_fname = obj_fname_list[plan_ind]
                start_pose = start_pose_list[plan_ind]
                real_start_pose = start_pose_list[plan_ind]
                experiment_manager.set_object_id(obj_id, obj_fname)

                # set as collision object all OTHER objects, other than what we are manipulating
                for idx in range(len(plan_total_list)):
                    if idx == plan_ind:
                        continue
                    # set as active object
                    action_planner.update_object(obj_id=obj_id_list[idx], mesh_file=obj_fname_list[idx])

                    # add to planning scene
                    action_planner.add_remove_scene_object('add')

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
                    if 'pull' in skeleton[i]:
                        palms[2] -= 0.002
                    palm_pose_plan.append(palms)

                # observe results
                full_plan = []
                for i in range(len(plan)):
                    if 'pull' in skeleton[i]:
                        local_plan = pulling_planning_wf(
                            util.list2pose_stamped(palm_pose_plan[i]),
                            util.list2pose_stamped(palm_pose_plan[i]),
                            util.pose_from_matrix(plan[i].transformation)
                        )
                    elif 'grasp' in skeleton[i]:
                        local_plan = grasp_planning_wf(
                            util.list2pose_stamped(palm_pose_plan[i][7:]),
                            util.list2pose_stamped(palm_pose_plan[i][:7]),
                            util.pose_from_matrix(plan[i].transformation)
                        )
                    full_plan.append(local_plan)


                grasp_success = True

                action_planner.active_arm = 'right'
                action_planner.inactive_arm = 'left'
                action_planner.update_object(obj_id=obj_id, mesh_file=obj_fname)

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

                embed()
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

                        # goal_obj_id2 = yumi_ar.pb_client.load_geom(
                        #     shape_type='mesh', 
                        #     visualfile=obj_fname, 
                        #     collifile=obj_fname, 
                        #     mesh_scale=[1.0]*3,
                        #     base_pos=[0.45, 0, 0.1], 
                        #     rgba=[0.0, 0.0, 0.95, 0.25],
                        #     mass=0.03)

                        # p.setCollisionFilterPair(goal_obj_id2, obj_id, -1, -1, enableCollision=False)
                        # p.setCollisionFilterPair(goal_obj_id2, goal_obj_id, -1, -1, enableCollision=False)
                        # cuboid_manager.filter_collisions(obj_id, goal_obj_id2)

                        for i, skill in enumerate(skeleton):
                        # for i, skill in enumerate(skeleton[2:]):
                        #     i += 2
                            # if i < len(skeleton) - 1:
                            #     yumi_ar.pb_client.reset_body(goal_obj_id2, pose_plan[i+1][0][:3], pose_plan[i+1][0][3:])
                            # else:
                            #     yumi_ar.pb_client.reset_body(goal_obj_id2, goal_pose[:3], goal_pose[3:])
                            if 'left' in skill:
                                arm = 'left'
                                action_planner.active_arm = 'left'
                                action_planner.inactive_arm = 'right'
                            else:
                                arm = 'right'
                                action_planner.active_arm = 'right'
                                action_planner.inactive_arm = 'left'
                            if 'pull' in skill:
                                action_planner.add_remove_scene_object('add')
                                time.sleep(0.5)
                                joints_r, joints_l = yumi_gs.move_to_joint_target_mp(cfg.RIGHT_INIT, cfg.LEFT_INIT)
                                for k in range(joints_r.shape[0]):
                                    jnts_r = joints_r[k, :]
                                    jnts_l = joints_l[k, :]
                                    yumi_ar.arm.set_jpos(jnts_r.tolist() + jnts_l.tolist(), wait=True)
                                    time.sleep(0.01)
                                action_planner.add_remove_scene_object('remove')                                
                                time.sleep(0.5)

                                _, _ = action_planner.single_arm_setup(full_plan[i][0], pre=True)
                                start_playback_time = time.time()
                                if not experiment_manager.still_pulling(arm=arm):
                                    while True:
                                        if experiment_manager.still_pulling(arm=arm) or time.time() - start_playback_time > 20.0:
                                            break
                                        action_planner.single_arm_approach(arm=arm)
                                        time.sleep(0.075)
                                        new_plan = pulling_planning_wf(
                                            yumi_gs.get_current_tip_poses()['left'],
                                            yumi_gs.get_current_tip_poses()['right'],
                                            util.pose_from_matrix(plan[i].transformation)
                                        )
                                    pull_plan = new_plan[0]
                                else:
                                    pull_plan = full_plan[i][0]
                                action_planner.playback_single_arm('pull', pull_plan, pre=False)
                                grasp_success = grasp_success and experiment_manager.still_pulling(n=False)
                                print('grasp success: ' + str(grasp_success))
                                time.sleep(0.5)
                                action_planner.single_arm_retract(arm=arm)

                            elif 'grasp' in skill:
                                action_planner.add_remove_scene_object('add')
                                time.sleep(0.5)
                                joints_r, joints_l = yumi_gs.move_to_joint_target_mp([0.9936, -2.1848, -0.9915, 0.8458, 3.7618,  1.5486,  0.1127],
                                                                                    [-1.0777, -2.1187, 0.995, 1.002,  -3.6834,  1.8132,  2.6405])
                                for k in range(joints_r.shape[0]):
                                    jnts_r = joints_r[k, :]
                                    jnts_l = joints_l[k, :]
                                    yumi_ar.arm.set_jpos(jnts_r.tolist() + jnts_l.tolist(), wait=True)
                                    time.sleep(0.01)
                                action_planner.add_remove_scene_object('remove')                                
                                time.sleep(0.5)
    
                                _, _ = action_planner.dual_arm_setup(full_plan[i][0], 0, pre=True)
                                start_playback_time = time.time()
                                if not experiment_manager.still_grasping():
                                    jj = 0
                                    while True:
                                        if experiment_manager.still_grasping() or time.time() - start_playback_time > 20.0:
                                            jj += 1
                                        if jj > 2:
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
                                    action_planner.playback_dual_arm('grasp', subplan, k, pre=False)
                                    if k == 1:
                                        grasp_success = grasp_success and experiment_manager.still_grasping(n=False)
                                        print('grasp success: ' + str(grasp_success))
                                    time.sleep(1.0)
                        action_planner.add_remove_scene_object('add')
                        time.sleep(0.5)
                        joints_r, joints_l = yumi_gs.move_to_joint_target_mp(cfg.RIGHT_OUT_OF_FRAME, cfg.LEFT_OUT_OF_FRAME)
                        for k in range(joints_r.shape[0]):
                            jnts_r = joints_r[k, :]
                            jnts_l = joints_l[k, :]
                            yumi_ar.arm.set_jpos(jnts_r.tolist() + jnts_l.tolist(), wait=True)
                            time.sleep(0.01)
                        action_planner.add_remove_scene_object('remove')                                       
                except ValueError as e:
                    print(e)
                    embed()
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

            yumi_ar.arm.go_home(ignore_physics=True)

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