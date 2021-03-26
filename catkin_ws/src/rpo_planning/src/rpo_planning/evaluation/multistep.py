import os
import os.path as osp
import sys
import time
import argparse
import numpy as np
import rospy
import rospkg
import signal
import threading
import pickle
import open3d
import copy
from random import shuffle

from airobot import Robot
from airobot.utils import common
from airobot import set_log_level, log_debug, log_info, log_warn, log_critical
import pybullet as p

from rpo_planning.utils import common as util
from rpo_planning.utils.object import CuboidSampler
from rpo_planning.utils.pb_visualize import GoalVisual
from rpo_planning.utils.visualize import PalmVis, PCDVis
from rpo_planning.utils.data import MultiBlockManager
from rpo_planning.utils import ros as simulation
from rpo_planning.utils.evaluate import GraspEvalManager
from rpo_planning.config.multistep_eval_cfg import get_multistep_cfg_defaults
from rpo_planning.config.explore_cfgs.default_skill_names import get_skillset_cfg
from rpo_planning.robot.multicam_env import YumiMulticamPybullet
from rpo_planning.execution.motion_playback import OpenLoopMacroActions
from rpo_planning.execution.closed_loop import ClosedLoopMacroActions
from rpo_planning.skills.mb_primitive_skills import SingleArmPrimitives, DualArmPrimitives 

from rpo_planning.skills.samplers.grasp import GraspSamplerBasic, GraspSamplerVAE
from rpo_planning.skills.samplers.pull import PullSamplerBasic, PullSamplerVAE
from rpo_planning.skills.samplers.push import PushSamplerBasic, PushSamplerVAE
from rpo_planning.motion_planning.primitive_planners import (
    pulling_planning_wf, grasp_planning_wf, pushing_planning_wf
)
from rpo_planning.skills.primitive_skills import (
    GraspSkill, PullLeftSkill, PullRightSkill, PushLeftSkill, PushRightSkill
)
from rpo_planning.utils.planning.pointcloud_plan import (
    PointCloudNode, PointCloudPlaneSegmentation
)
from rpo_planning.pointcloud_planning.rpo_planner import PointCloudTree
from rpo_planning.utils.exceptions import (
    SkillApproachError, InverseKinematicsError, 
    DualArmAlignmentError, PlanWaypointsError, 
    MoveToJointTargetError
)
from rpo_planning.utils.planning.rpo_plan_visualize import FloatingPalmPlanVisualizer


def main(args):
    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    rospack = rospkg.RosPack()
    skill_config_path = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/config/skill_cfgs')
    pull_cfg_file = osp.join(skill_config_path, 'pull') + ".yaml"
    pull_cfg = get_multistep_cfg_defaults()
    pull_cfg.merge_from_file(pull_cfg_file)
    pull_cfg.freeze()

    grasp_cfg_file = osp.join(skill_config_path, 'grasp') + ".yaml"
    grasp_cfg = get_multistep_cfg_defaults()
    grasp_cfg.merge_from_file(grasp_cfg_file)
    grasp_cfg.freeze()

    push_cfg_file = osp.join(skill_config_path, 'push') + ".yaml"
    push_cfg = get_multistep_cfg_defaults()
    push_cfg.merge_from_file(push_cfg_file)
    push_cfg.freeze()

    cfg = pull_cfg
    skillset_cfg = get_skillset_cfg()
    rospy.init_node('EvalMultistep')
    signal.signal(signal.SIGINT, util.signal_handler)

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
        lateralFriction=0.5,
        restitution=restitution,
        contactStiffness=K,
        contactDamping=alpha*K,
        rollingFriction=args.rolling
    )

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        l_gel_id,
        lateralFriction=0.5,
        restitution=restitution,
        contactStiffness=K,
        contactDamping=alpha*K,
        rollingFriction=args.rolling
    )

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        table_id,
        lateralFriction=0.1
    )

    yumi_gs = YumiMulticamPybullet(
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
        # cuboid_fname = 'catkin_ws/src/config/descriptions/meshes/objects/cuboids/test_cuboid_smaller_4479.stl'
    else:
        cuboid_fname = osp.join(
            os.environ['CODE_BASE'],
            args.config_package_path + 'descriptions/meshes/objects/' + \
            args.object_name + '_experiments.stl')
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

    if goal_visualization:
        trans_box_lock = threading.RLock()
        goal_viz = GoalVisual(
            trans_box_lock,
            goal_obj_id,
            yumi_ar.pb_client.get_client_id(),
            cfg.OBJECT_POSE_3,
            show_init=False)

    p.changeDynamics(
        obj_id,
        -1,
        lateralFriction=1.0
    )

    goal_faces = [0, 1, 2, 3, 4, 5]
    shuffle(goal_faces)
    goal_face = goal_faces[0]

    exp_single = SingleArmPrimitives(
        cfg,
        yumi_ar.pb_client.get_client_id(),
        obj_id,
        cuboid_fname)

    action_planner = ClosedLoopMacroActions(
        cfg,
        yumi_gs,
        obj_id,
        yumi_ar.pb_client.get_client_id(),
        osp.join(os.environ['CODE_BASE'], args.config_package_path),
        replan=args.replan,
        object_mesh_file=mesh_file
    )

    action_planner.update_object(obj_id, mesh_file)

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

    yumi_ar.pb_client.remove_body(obj_id)
    if goal_visualization:
        yumi_ar.pb_client.remove_body(goal_obj_id)

    target_surface_skeleton = None

    if args.skeleton == 'pg':
        skeleton = ['pull_right', 'grasp']
    elif args.skeleton == 'gp':
        skeleton = ['grasp', 'pull_right']
    elif args.skeleton == 'pgp':
        skeleton = ['pull_right', 'grasp', 'pull_right']
    else:
        raise ValueError('Unrecognized plan skeleton!')

    if args.bookshelf and args.demo:
        obs, pcd = yumi_gs.get_observation(
            obj_id=obj_id,
            robot_table_id=(yumi_ar.arm.robot_id, 28))
        shelf_pcd = open3d.geometry.PointCloud()
        shelf_pcd.points = open3d.utility.Vector3dVector(np.concatenate(obs['table_pcd_pts']))
        shelf_pointcloud = np.asarray(shelf_pcd.points)
        z_sort = np.sort(shelf_pointcloud[:, 2])[::-1]
        top_z_2 = z_sort[10]
        shelf_target_surface = shelf_pointcloud[np.where(shelf_pointcloud[:, 2] > 0.9*top_z_2)[0], :]
        target_surface_skeleton = [None, None, None, shelf_target_surface, shelf_target_surface]
        skeleton = ['pull_right', 'grasp', 'pull_right', 'grasp_pp', 'pull_left']

    if args.baseline:
        log_debug('LOADING BASELINE SAMPLERS')
        pull_sampler = PullSamplerBasic()
        grasp_sampler = GraspSamplerBasic(None)
        push_sampler = PushSamplerVAE()
    else:
        log_debug('LOADING LEARNED SAMPLERS')
        pull_sampler = PullSamplerVAE(sampler_prefix='pull_0_vae_')
        push_sampler = PushSamplerVAE(sampler_prefix='push_0_vae_')
        grasp_sampler = GraspSamplerVAE(sampler_prefix='grasp_0_vae_', default_target=None)

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

    push_left_skill = PushLeftSkill(
        push_sampler,
        yumi_gs,
        pushing_planning_wf,
        ignore_mp=False,
        avoid_collisions=True
    )

    grasp_skill = GraspSkill(grasp_sampler, yumi_gs, grasp_planning_wf)
    grasp_pp_skill = GraspSkill(grasp_sampler, yumi_gs, grasp_planning_wf, pp=True)

    skills = {}
    for name in skillset_cfg.SKILL_NAMES:
        skills[name] = None
    skills['pull_right'] = pull_right_skill
    skills['pull_left'] = pull_left_skill
    skills['grasp'] = grasp_skill
    skills['grasp_pp'] = grasp_pp_skill
    skills['push_right'] = push_right_skill
    skills['push_left'] = push_left_skill

    # set up rpo plan visualizer
    rpo_plan_viz = FloatingPalmPlanVisualizer(palm_mesh_file, table_mesh_file, cfg, skills)

    if args.demo_type == 'cuboid_regular' and not args.bookshelf:
        problems_file = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning', args.planning_problems_dir, 'test_problems_0/demo_0_formatted_half_minimal.pkl')
        # problems_file = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning', args.planning_problems_dir, 'test_problems_0/demo_0_formatted_minimal.pkl')
        # problems_file = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning', args.planning_problems_dir, 'test_problems_0/demo_1_formatted_minimal.pkl')
    elif args.demo_type == 'cuboid_bookshelf' and args.bookshelf:
        problems_file = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning', args.planning_problems_dir, 'bookshelf_cuboid/bookshelf_problems_formatted.pkl')
    elif args.demo_type == 'bookshelf' and args.bookshelf:
        problems_file = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning', args.planning_problems_dir, 'bookshelf_1/bookshelf_problems_formatted.pkl')
    elif args.demo_type == 'general':
        problems_file = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning', args.planning_problems_dir, 'gen_obj_1/gen_obj_problems_0_formatted.pkl')
    else:
        raise ValueError('Demo type not recognized')

    # problems_file = osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives/planning_problems/stacking_cuboids_0/stacking_cuboids_problems_0_formatted.pkl')
    with open(problems_file, 'rb') as f:
        problems_data = pickle.load(f)

    prob_inds = np.arange(len(problems_data), dtype=np.int64).tolist()
    data_inds = np.arange(len(problems_data[0]['problems']), dtype=np.int64).tolist()

    experiment_manager = GraspEvalManager(
        yumi_gs,
        yumi_ar.pb_client.get_client_id(),
        pickle_path,
        args.exp_name,
        cfg
    )

    # check number of problems we have gone through, and keep looping until we get to the target
    total_number_problems = len(problems_data) * len(problems_data[0]['problems'])
    if args.save_data:
        number_problems_so_far = len(os.listdir(pickle_path))
    else:
        number_problems_so_far = 0

    total_trial_number = 0
    while number_problems_so_far < total_number_problems:
        if args.save_data:
            number_problems_so_far = len(os.listdir(pickle_path))
        else:
            number_problems_so_far = 0
        print('Problems so far / Total : %d / %d' % (number_problems_so_far, total_number_problems))
        total_trial_number += 1
        # prob_ind = 8
        # data_ind = 15

        # ### intro figure data:
        # obj_fname = osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/config/descriptions/meshes/objects/cuboids/test_cuboid_smaller_4867.stl')
        # ###

        prob_ind = prob_inds[np.random.randint(len(prob_inds))]
        data_ind = data_inds[np.random.randint(len(data_inds))]

        problem_data = problems_data[prob_ind]['problems'][data_ind]
        stl_file = problems_data[prob_ind]['object_name'].split('catkin_ws/')[1]
        obj_fname = osp.join(os.environ['CODE_BASE'], 'catkin_ws', stl_file)
        obj_name = obj_fname.split('.stl')[0].split('/meshes/objects/cuboids/')[1]
        scale = problems_data[prob_ind]['object_scale']
        start_pose = problem_data['start_vis'].tolist()
        goal_pose = problem_data['goal_vis'].tolist()
        transformation_des = problem_data['transformation']

        obj_data_fname = osp.join(
            pickle_path,
            obj_name+'_'+str(prob_ind)+'_'+str(data_ind)+'_ms_eval_data.pkl')
        if osp.exists(obj_data_fname):
            print('already ran this trial, moving to next problem')
            continue


        # ### use this to put the goal state in the left back corner of the table
        # T = util.rand_body_yaw_transform(orig_goal_pose[:3], min_theta=np.pi/2, max_theta=3*np.pi/2)
        # T_t = np.eye(4)
        # # T_t[0, -1] = 0.2 - start_pose[0]
        # # T_t[1, -1] = 0.4 - start_pose[1]
        # T_t[0, -1] = 0.2 - orig_goal_pose[0]
        # T_t[1, -1] = 0.4 - orig_goal_pose[1]
        # # transformation_des = np.matmul(T_t, T)
        # transformation_des = np.matmul(T_t, np.matmul(T, orig_transformation_des))   # to keep whatever initial reorientation
        # # transformation_des = np.matmul(T_t, T)   # to keep in the plane
        # goal_pose = util.pose_stamped2list(util.transform_pose(
        #     util.list2pose_stamped(start_pose),
        #     util.pose_from_matrix(transformation_des)))

        # ### use this to put the start state on the right/left side of the table
        # start_x = np.random.random() * (0.4 - 0.2) + 0.2
        # start_y = - (np.random.random() * (0.4 - 0.3) + 0.3)
        # start_pose = [start_x, start_y] + orig_start_pose[2:]

        # transformation_des = util.matrix_from_pose(
        #     util.get_transform(util.list2pose_stamped(goal_pose), util.list2pose_stamped(start_pose))
        # )


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

        if goal_visualization:
            goal_viz.update_goal_obj(goal_obj_id)
            goal_viz.update_goal_state(goal_pose)
            goal_viz.hide_goal_obj()
            cuboid_manager.filter_collisions(obj_id, goal_obj_id)

            time.sleep(1.0)


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
        time.sleep(0.5)

        real_start_pos = p.getBasePositionAndOrientation(obj_id)[0]
        real_start_ori = p.getBasePositionAndOrientation(obj_id)[1]
        real_start_pose = list(real_start_pos) + list(real_start_ori)

        if goal_visualization:
            real_goal_pos = p.getBasePositionAndOrientation(goal_obj_id)[0]
            real_goal_ori = p.getBasePositionAndOrientation(goal_obj_id)[1]
            real_goal_pose = list(real_goal_pos) + list(real_goal_ori)

            transformation_des = util.matrix_from_pose(
                util.get_transform(util.list2pose_stamped(real_goal_pose), util.list2pose_stamped(real_start_pose))
            )

            goal_pose = real_goal_pose

        if args.demo:
            pass
        else:
            if args.skeleton == 'pgp':
                # search for a random final SE(2) transformation after the nominal goal pose from the nominal problem
                while True:
                    T = exp_single.get_rand_trans_yaw_T(pos=goal_pose[:3])
                    new_goal_pose = util.transform_pose(util.list2pose_stamped(goal_pose), util.pose_from_matrix(T))
                    goal_pose_2_list = util.pose_stamped2np(new_goal_pose)

                    if goal_pose_2_list[0] > 0.2 and goal_pose_2_list[0] < 0.4 and \
                            goal_pose_2_list[1] > -0.3 and goal_pose_2_list[1] < 0.1:
                        break

                goal_pose = goal_pose_2_list
                transformation_des = np.matmul(T, transformation_des)

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
            robot_table_id=(yumi_ar.arm.robot_id, table_id),
            cam_inds=args.camera_inds,
            depth_noise=args.pcd_noise,
            depth_noise_std=args.pcd_noise_std,
            depth_noise_rate=args.pcd_noise_rate)

        if goal_visualization:
            goal_viz.update_goal_state(goal_pose)
            goal_viz.show_goal_obj()

        pointcloud_pts = np.asarray(obs['down_pcd_pts'][:100, :], dtype=np.float32)
        pointcloud_pts_full = np.asarray(np.concatenate(obs['pcd_pts']), dtype=np.float32)
        table_pts = np.concatenate(obs['table_pcd_pts'], axis=0)[::500, :]
        grasp_sampler.update_default_target(table_pts)


        trial_data = {}
        trial_data['start_pcd'] = pointcloud_pts_full
        trial_data['start_pcd_down'] = pointcloud_pts
        trial_data['obj_fname'] = cuboid_fname
        trial_data['start_pose'] = np.asarray(real_start_pose)
        trial_data['goal_pose'] = np.asarray(goal_pose)
        trial_data['goal_pose_global'] = np.asarray(goal_pose)
        trial_data['trans_des_global'] = transformation_des

        trial_data['skeleton'] = skeleton

        trial_data['predictions'] = {}
        model_path1 = grasp_sampler.get_model_path()
        model_path2 = pull_sampler.get_model_path()
        model_path3 = push_sampler.get_model_path()
        trial_data['predictions']['model_path'] = [model_path1, model_path2, model_path3]

        # save current camera information
        trial_data['camera_inds'] = args.camera_inds
        trial_data['camera_noise'] = None
        if args.pcd_noise:
            trial_data['camera_noise'] = {}
            trial_data['camera_noise']['std'] = args.pcd_noise_std
            trial_data['camera_noise']['rate'] = args.pcd_noise_rate

        # only use start/goal feasibility check for baseline
        start_goal_feasibility = args.baseline
        failure_tracking = args.failure_tracking
        print('Start/Goal Feasibility Checking: ' + str(start_goal_feasibility))
        # plan!
        planner = PointCloudTree(
            pointcloud_pts,
            transformation_des,
            skeleton,
            skills,
            max_steps=args.max_steps,
            start_pcd_full=pointcloud_pts_full,
            visualize=True,
            obj_id=goal_obj_id,
            start_pose=util.list2pose_stamped(start_pose),
            target_surface_pcds=target_surface_skeleton,
            start_goal_palm_check=start_goal_feasibility,
            tracking_failures=failure_tracking,
            timeout=30)
        start_plan_time = time.time()

        # plan, will return None if no plan has been found before timeout
        if args.no_skeleton:
            plan_total = planner.plan_max_length()
            skeleton = []
            if plan_total is not None:
                for node in plan_total:
                    if node.skill is not None:
                        skeleton.append(node.skill)
        else:
            plan_total = planner.plan()

        trial_data['planning_failure'] = planner.planning_stat_tracker.collect_data()
        string = ''
        string = string + 'total_samples: %d, ' % trial_data['planning_failure']['total_samples']
        for k, v in trial_data['planning_failure']['skeleton_samples'].items():
            string = string + ' %s: %d, ' %(k, v)
        string = string + '\n\n\n'
        print(string)

        if plan_total is None:
            print('Could not find plan')
            experiment_manager.set_mp_success(False, 1)
            experiment_manager.set_planning_failure(trial_data['planning_failure'])
            obj_data = experiment_manager.get_object_data()
            if args.save_data:
                print('Saving to: ' + str(obj_data_fname))
                with open(obj_data_fname, 'wb') as f:
                    pickle.dump(obj_data, f)
            continue

        plan = copy.deepcopy(plan_total[1:])
        # rpo_plan_imgs = rpo_plan_viz.render_plan(skeleton, plan_total, table_pts)

        trial_data['planning_time'] = time.time() - start_plan_time

        if args.trimesh_viz:
            ind = 2
            pcd_data = copy.deepcopy(problem_data)
            pcd_data['start'] = plan_total[ind].pointcloud_full
            pcd_data['object_pointcloud'] = plan_total[ind].pointcloud_full
            pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(plan_total[ind+1].transformation)))
            pcd_data['contact_world_frame_right'] = np.asarray(plan_total[ind+1].palms[:7])
            if 'pull' in skeleton[ind]:
                pcd_data['contact_world_frame_left'] = np.asarray(plan_total[ind+1].palms[:7])
            else:
                pcd_data['contact_world_frame_left'] = np.asarray(plan_total[ind+1].palms[7:])
            scene = viz_palms.vis_palms_pcd(pcd_data, world=True, centered=False, corr=False)
            scene.show()

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
            if 'pull' in skeleton[i]:
                palms[2] -= 0.0015
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
            elif 'push' in skeleton[i]:
                local_plan = pushing_planning_wf(
                    util.list2pose_stamped(palm_pose_plan[i]),
                    util.list2pose_stamped(palm_pose_plan[i]),
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

        real_start_pos = p.getBasePositionAndOrientation(obj_id)[0]
        real_start_ori = p.getBasePositionAndOrientation(obj_id)[1]
        real_start_pose = list(real_start_pos) + list(real_start_ori)
        real_start_mat = util.matrix_from_pose(util.list2pose_stamped(real_start_pose))

        goal_obj_id2 = None
        try:
            start_playback_time = time.time()
            for playback in range(args.playback_num):
                yumi_ar.arm.go_home(ignore_physics=True)
                try:
                    yumi_ar.pb_client.remove_body(goal_obj_id2)
                except:
                    pass
                if playback > 0 and goal_visualization:
                    goal_viz.hide_goal_obj()
                if playback == 2:
                    goal_viz.show_goal_obj()
                    goal_obj_id2 = yumi_ar.pb_client.load_geom(
                        shape_type='mesh',
                        visualfile=obj_fname,
                        collifile=obj_fname,
                        mesh_scale=[1.0]*3,
                        base_pos=[0.45, 0, 0.1],
                        rgba=[0.0, 0.0, 0.95, 0.25],
                        mass=0.03)

                    p.setCollisionFilterPair(goal_obj_id2, obj_id, -1, -1, enableCollision=False)
                    p.setCollisionFilterPair(goal_obj_id2, goal_obj_id, -1, -1, enableCollision=False)
                    cuboid_manager.filter_collisions(obj_id, goal_obj_id2)

                yumi_ar.pb_client.reset_body(obj_id, pose_plan[0][0][:3], pose_plan[0][0][3:])
                p.changeDynamics(
                    obj_id,
                    -1,
                    lateralFriction=1.0
                )

                for i, skill in enumerate(skeleton):
                    if playback == 2:
                        if i < len(skeleton) - 1:
                            yumi_ar.pb_client.reset_body(goal_obj_id2, pose_plan[i+1][0][:3], pose_plan[i+1][0][3:])
                        else:
                            yumi_ar.pb_client.reset_body(goal_obj_id2, goal_pose[:3], goal_pose[3:])

                    if 'left' in skill:
                        arm = 'left'
                        action_planner.active_arm = 'left'
                        action_planner.inactive_arm = 'right'
                    else:
                        arm = 'right'
                        action_planner.active_arm = 'right'
                        action_planner.inactive_arm = 'left'
                    if 'push' in skill:
                        p.changeDynamics(
                            yumi_ar.arm.robot_id,
                            r_gel_id,
                            rollingFriction=1.0
                        )

                        p.changeDynamics(
                            yumi_ar.arm.robot_id,
                            l_gel_id,
                            rollingFriction=1.0
                        )
                    else:
                        p.changeDynamics(
                            yumi_ar.arm.robot_id,
                            r_gel_id,
                            rollingFriction=1e-4
                        )

                        p.changeDynamics(
                            yumi_ar.arm.robot_id,
                            l_gel_id,
                            rollingFriction=1e-4
                        )
                    if 'pull' in skill or 'push' in skill:
                        skill_cfg = pull_cfg if 'pull' in skill else push_cfg
                        # set arm configuration to good start state
                        action_planner.add_remove_scene_object('add')
                        cuboid_manager.robot_collisions_filter(obj_id, enable=False)
                        time.sleep(0.5)
                        if args.ignore_physics:
                            yumi_ar.arm.set_jpos(skill_cfg.RIGHT_INIT + skill_cfg.LEFT_INIT,
                                                ignore_physics=True)
                        else:
                            _, _ = yumi_gs.move_to_joint_target_mp(skill_cfg.RIGHT_INIT, skill_cfg.LEFT_INIT,
                                                                    execute=True)
                        action_planner.add_remove_scene_object('remove')
                        cuboid_manager.robot_collisions_filter(obj_id, enable=True)
                        time.sleep(0.5)

                        # move to making contact, and ensure contact is made
                        _, _ = action_planner.single_arm_setup(full_plan[i][0], pre=True)
                        start_playback_time = time.time()
                        n = True if 'pull' in skill else False
                        if not experiment_manager.still_pulling(arm=arm, n=n):
                            while True:
                                if experiment_manager.still_pulling(arm=arm, n=n) or time.time() - start_playback_time > 20.0:
                                    break
                                action_planner.single_arm_approach(arm=arm)
                                time.sleep(0.075)
                                if 'pull' in skill:
                                    new_plan = pulling_planning_wf(
                                        yumi_gs.get_current_tip_poses()['left'],
                                        yumi_gs.get_current_tip_poses()['right'],
                                        util.pose_from_matrix(plan[i].transformation)
                                    )
                                else:
                                    new_plan = pushing_planning_wf(
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
                        cuboid_manager.robot_collisions_filter(obj_id, enable=False)
                        time.sleep(0.5)
                        if args.ignore_physics:
                            yumi_ar.arm.set_jpos(grasp_cfg.RIGHT_INIT + grasp_cfg.LEFT_INIT,
                                                ignore_physics=True)
                        else:
                            _, _ = yumi_gs.move_to_joint_target_mp(grasp_cfg.RIGHT_INIT, grasp_cfg.LEFT_INIT,
                                                                    execute=True)
                        action_planner.add_remove_scene_object('remove')
                        cuboid_manager.robot_collisions_filter(obj_id, enable=True)
                        time.sleep(0.5)

                        _, _ = action_planner.dual_arm_setup(full_plan[i][0], 0, pre=True)
                        action_planner.add_remove_scene_object('remove')
                        start_playback_time = time.time()
                        if not experiment_manager.still_grasping():
                            jj = 0
                            while True:
                                if experiment_manager.still_grasping() or time.time() - start_playback_time > 20.0:
                                    jj += 1
                                if jj > 2:
                                    break
                                # trying to catch IK breaking on the guarded approach.
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
        # except (ValueError, TypeError) as e:  # what was the TypeError for?
        except (SkillApproachError, InverseKinematicsError,
                PlanWaypointsError, DualArmAlignmentError,
                MoveToJointTargetError) as e:
            log_warn(e)
            experiment_manager.set_mp_success(True, 1)
            experiment_manager.set_planning_failure(trial_data['planning_failure'])
            experiment_manager.set_execute_success(False)
            obj_data = experiment_manager.get_object_data()
            if args.save_data:
                print('Saving to: ' + str(obj_data_fname))
                with open(obj_data_fname, 'wb') as f:
                    pickle.dump(obj_data, f)
            try:
                yumi_ar.pb_client.remove_body(goal_obj_id2)
            except:
                pass
            continue

        if not grasp_success:
            print('failed grasp success!')
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
        # kvs['grasp_success'] = obj_data['grasp_success']
        # kvs['pos_err'] = np.mean(obj_data['final_pos_error'])
        # kvs['ori_err'] = np.mean(obj_data['final_ori_error'])
        kvs['grasp_success'] = sum(obj_data['grasp_success']) * 100.0 / obj_data['trials']
        kvs['pos_err (filtered)'] = np.mean(obj_data['final_pos_error_filtered'])
        kvs['ori_err (filtered)'] = np.mean(obj_data['final_ori_error_filtered'])
        kvs['planning_time'] = obj_data['planning_time']
        string = ''

        for k, v in kvs.items():
            string += "%s: %.3f, " % (k,v)
        print(string)

        if args.save_data:
            print('Saving to: ' + str(obj_data_fname))
            with open(obj_data_fname, 'wb') as f:
                pickle.dump(obj_data, f)
        try:
            yumi_ar.pb_client.remove_body(goal_obj_id2)
        except:
            pass

        yumi_ar.arm.go_home(ignore_physics=True)


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
        default='catkin_ws/src/config/')

    parser.add_argument(
        '--planning_problems_dir',
        type=str,
        default='data/planning_problems')

    parser.add_argument(
        '--example_config_path',
        type=str,
        default='catkin_ws/src/primitives/config')

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

    parser.add_argument(
        '--no_skeleton', action='store_true'
    )

    parser.add_argument(
        '--max_steps', type=int, default=5
    )

    parser.add_argument(
        '--ignore_physics', action='store_true'
    )

    parser.add_argument(
        '--baseline', action='store_true'
    )

    parser.add_argument(
        '--camera_inds', nargs='+', type=int
    )

    parser.add_argument(
        '--pcd_noise', action='store_true'
    )

    parser.add_argument(
        '--pcd_noise_std', type=float, default=0.0025
    )

    parser.add_argument(
        '--pcd_noise_rate', type=float, default=0.00025
    )

    parser.add_argument(
        '--failure_tracking', action='store_true'
    )

    parser.add_argument(
        '--demo', action='store_true'
    )

    parser.add_argument(
        '--bookshelf', action='store_true'
    )

    parser.add_argument(
        '--demo_type', type=str, default='cuboid_regular'
    )

    args = parser.parse_args()
    main(args)
