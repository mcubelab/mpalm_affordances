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
import random
import lcm
from yacs.config import CfgNode as CN

from airobot import Robot
from airobot.utils import common
from airobot import set_log_level, log_debug, log_info, log_warn, log_critical
import pybullet as p

from rpo_planning.utils import common as util
from rpo_planning.execution.motion_playback import OpenLoopMacroActions
from rpo_planning.config.base_skill_cfg import get_skill_cfg_defaults
from rpo_planning.config.explore_task_cfg import get_task_cfg_defaults
from rpo_planning.config.multistep_eval_cfg import get_multistep_cfg_defaults
from rpo_planning.config.explore_cfgs.default_skill_names import get_skillset_cfg
from rpo_planning.robot.multicam_env import YumiMulticamPybullet 
from rpo_planning.utils.object import CuboidSampler
from rpo_planning.utils.pb_visualize import GoalVisual
from rpo_planning.utils.data import MultiBlockManager
from rpo_planning.utils.motion import guard
from rpo_planning.utils.planning.pointcloud_plan import PointCloudNode
from rpo_planning.utils.planning.skeleton import SkillSurfaceSkeleton
from rpo_planning.utils.visualize import PCDVis, PalmVis
from rpo_planning.utils.exploration.task_sampler import TaskSampler
from rpo_planning.utils.exploration.skeleton_processor import (
    process_skeleleton_prediction, separate_skills_and_surfaces)
from rpo_planning.utils.exploration.replay_data import rpo_plan2lcm
from rpo_planning.utils.exploration.client_init import PlanningClientInit, PlanningClientRPC

from rpo_planning.skills.samplers.pull import PullSamplerBasic, PullSamplerVAE
from rpo_planning.skills.samplers.push import PushSamplerBasic, PushSamplerVAE
from rpo_planning.skills.samplers.grasp import GraspSamplerBasic, GraspSamplerVAE
from rpo_planning.skills.primitive_skills import (
    GraspSkill, PullRightSkill, PullLeftSkill, PushRightSkill, PushLeftSkill
)
from rpo_planning.motion_planning.primitive_planners import (
    grasp_planning_wf, pulling_planning_wf, pushing_planning_wf
)
from rpo_planning.exploration.environment.play_env import PlayEnvironment, PlayObjects
from rpo_planning.exploration.skeleton_sampler import SkeletonSampler
from rpo_planning.pointcloud_planning.rpo_learner import PointCloudTreeLearner
from rpo_planning.utils.planning.rpo_plan_visualize import FloatingPalmPlanVisualizer
from rpo_planning.evaluation.exploration.experiment_manager import SimpleRPOEvalManager
# from replay_buffer import TransitionBuffer


class SkillExplorer(object):
    def __init__(self, skills):
        self.skills = skills

    def sample_skill(self, strategy=None):
        return np.random.choice(self.skills.keys(), 1)


def main(args):
    set_log_level('debug')

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

    rospy.init_node('ExploreSkeletonSingle')
    skillset_cfg = get_skillset_cfg()
    signal.signal(signal.SIGINT, util.signal_handler)
    client_rpc = PlanningClientRPC()
    skill2index = client_rpc.get_skill2index()    
    exp_name = client_rpc.get_experiment_name()
    experiment_cfg = CN(client_rpc.get_experiment_config())
    train_args = client_rpc.get_train_args()

    np.random.seed(args.np_seed)
    save_results_dir = osp.join(
        rospack.get_path('rpo_planning'), 'src/rpo_planning', args.data_dir, args.save_rl_data_dir, exp_name 
    )
    if not osp.exists(save_results_dir):
        os.makedirs(save_results_dir)
    log_info('Explore skeleton main: Saving results to directory: %s' % save_results_dir)

    np.random.seed(args.np_seed)
    # initialize airobot and modify dynamics
    yumi_ar = Robot('yumi_palms',
                    pb=True,
                    pb_cfg={'gui': args.visualize,
                            'opengl_render': False},
                    arm_cfg={'self_collision': False,
                             'seed': args.np_seed})

    yumi_ar.arm.go_home(ignore_physics=True)

    yumi_gs = YumiMulticamPybullet(
        yumi_ar,
        cfg,
        exec_thread=False,
    )

    if args.sim:
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

    skill_ignore_mp = True
    pull_right_skill = PullRightSkill(
        pull_sampler,
        yumi_gs,
        pulling_planning_wf,
        ignore_mp=skill_ignore_mp,
        avoid_collisions=True
    )

    pull_left_skill = PullLeftSkill(
        pull_sampler,
        yumi_gs,
        pulling_planning_wf,
        ignore_mp=skill_ignore_mp,
        avoid_collisions=True
    )

    push_right_skill = PushRightSkill(
        push_sampler,
        yumi_gs,
        pushing_planning_wf,
        ignore_mp=skill_ignore_mp,
        avoid_collisions=True
    )

    push_left_skill = PushLeftSkill(
        push_sampler,
        yumi_gs,
        pushing_planning_wf,
        ignore_mp=skill_ignore_mp,
        avoid_collisions=True
    )

    grasp_skill = GraspSkill(grasp_sampler, yumi_gs, grasp_planning_wf, ignore_mp=skill_ignore_mp, avoid_collisions=True)
    grasp_pp_skill = GraspSkill(grasp_sampler, yumi_gs, grasp_planning_wf, ignore_mp=skill_ignore_mp, avoid_collisions=True, pp=True)

    skills = {}
    for name in skillset_cfg.SKILL_NAMES:
        skills[name] = None
    skills['pull_right'] = pull_right_skill
    skills['pull_left'] = pull_left_skill
    skills['grasp'] = grasp_skill
    skills['grasp_pp'] = grasp_pp_skill
    skills['push_right'] = push_right_skill
    skills['push_left'] = push_left_skill

    # create exploring agent
    agent = SkillExplorer(skills)

    pb_info = None
    if args.sim:
        pb_info = {}
        pb_info['object_id'] = None
        pb_info['object_mesh_file'] = None
        pb_info['pb_client'] = yumi_ar.pb_client.get_client_id()

    action_planner = OpenLoopMacroActions(
        cfg,
        yumi_gs,
        pb=args.sim,
        pb_info=pb_info
    )

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
        table_id=cfg.TABLE_ID,
        r_gel_id=cfg.RIGHT_GEL_ID,
        l_gel_id=cfg.LEFT_GEL_ID)

    # visualization stuff
    palm_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.PALM_MESH_FILE)
    table_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.TABLE_MESH_FILE)
    viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
    rpo_plan_viz = FloatingPalmPlanVisualizer(palm_mesh_file, table_mesh_file, cfg, skills)

    # create interface to make guarded movements
    guarder = guard.GuardedMover(robot=yumi_gs, pb_client=yumi_ar.pb_client.get_client_id(), cfg=cfg)

    # create manager
    env = PlayEnvironment(pb_client=yumi_ar.pb_client, robot=yumi_gs, cuboid_manager=cuboid_manager, cuboid_sampler=cuboid_sampler, target_surface_ids=[cfg.TABLE_ID])

    # sample some objects
    env.sample_objects(n=1)

    # instantiate objects in the world at specific poses
    env.initialize_object_states()
    # create task sampler
    task_cfg_file = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/config/task_cfgs', args.task_config_file + '.yaml')
    task_cfg = get_task_cfg_defaults()
    task_cfg.merge_from_file(task_cfg_file)
    task_cfg.freeze()
    task_sampler = TaskSampler(
        osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning', args.task_problems_path), 
        task_cfg)

    # create replay buffer
    # experience_buffer = TransitionBuffer(size, observation_n, action_n, device, goal_n=None)
    
    # interface to obtaining skeleton predictions from the NN (handles LCM message passing and skeleton processing)
    skeleton_policy = SkeletonSampler()
    task_difficulty = args.task_difficulty

    # experiment data management
    skeleton_data_dir = save_results_dir
    skeleton_exp_name = exp_name
    experiment_manager = SimpleRPOEvalManager(skeleton_data_dir, skeleton_exp_name, cfg)
    log_info('RPO Skeleton Explore: Saving results to directory: %s' % skeleton_data_dir)
    print(experiment_cfg)

    assert experiment_cfg.exploration.mode in ['epsilon_greedy_decay', 'constant', 'none'], 'Unrecognized exploration strategy'
    
    if experiment_cfg.exploration.mode == 'none':
        epsilon = 0
    else:
        epsilon = experiment_cfg.exploration.start_epsilon
        if experiment_cfg.exploration.mode == 'epsilon_greedy_decay':
            epsilon_rate = experiment_cfg.exploration.decay_rate
        elif experiment_cfg.exploration.mode == 'constant':
            epsilon_rate = 1.0
    epsilon_info_str = 'RPO Exploration Main: Exploration epsilon start value: %.3f, epsilon decay rate: %.3f' % (epsilon, epsilon_rate)
    log_info(epsilon_info_str)

    # write all experiment configuration
    master_config_dict = {}
    master_config_dict['experiment_cfg'] = util.cn2dict(experiment_cfg)
    master_config_dict['train_args'] = train_args
    master_config_dict['plan_args'] = args.__dict__
    master_config_dict['task_cfg'] = util.cn2dict(task_cfg)
    with open(osp.join(save_results_dir, 'master_config.pkl'), 'wb') as f:
        pickle.dump(master_config_dict, f)

    trials = 1 
    while True:
        # sample a task
        start_pose, goal_pose, fname, pointcloud, transformation_des, surfaces, task_surfaces, scene_pcd = task_sampler.sample_full(task_difficulty)
        pointcloud_sparse = pointcloud[::int(pointcloud.shape[0]/100), :][:100]
        scene_pcd = scene_pcd[::int(scene_pcd.shape[0]/100), :][:100]
        # scene_pointcloud = np.concatenate(surfaces.values())

        # run the policy to get a skeleton
        predicted_skeleton, predicted_inds = skeleton_policy.predict(pointcloud_sparse, scene_pcd, transformation_des)
        # log_debug('predicted: ', predicted_skeleton)

        _, predicted_surfaces = separate_skills_and_surfaces(predicted_skeleton, skillset_cfg)

        target_surfaces = []
        for surface_name in predicted_surfaces:
            surface_pcd = surfaces[surface_name]
            target_surfaces.append(surface_pcd)

        # create unified skeleton object (contains all info about target point clouds and skeleton for planner)
        plan_skeleton = SkillSurfaceSkeleton(
            predicted_skeleton,
            predicted_inds,
            skillset_cfg=skillset_cfg,
            skill2index=skill2index,
            skeleton_surface_pcds=target_surfaces
        )

        if trials % args.eval_freq == 0:
            evaluation = True
            epsilon_to_use = 0.0
            timeout_to_use = args.eval_timeout
            print('Eval skeleton: ', plan_skeleton.skeleton_full)
        else:
            evaluation = False
            epsilon_to_use = epsilon
            timeout_to_use = args.timeout
        planner = PointCloudTreeLearner(
            start_pcd=pointcloud_sparse,
            start_pcd_full=pointcloud,
            trans_des=transformation_des,
            plan_skeleton=plan_skeleton,
            skills=skills,
            max_steps=args.max_steps,
            skeleton_policy=None,
            skillset_cfg=skillset_cfg,
            pointcloud_surfaces=surfaces,
            motion_planning=True,
            timeout=timeout_to_use,
            epsilon=epsilon_to_use
        )
        epsilon = epsilon * epsilon_rate

        # log_debug('Planning from RPO_Planning worker ID: %d' % worker_id)
        plan_total = planner.plan_with_skeleton_explore(just_explore=True)        

        print('done planning')

        if plan_total is not None:
            real_skeleton = []
            for i in range(1, len(plan_total)):
                real_skeleton.append(plan_total[i].skill)
            skeleton = real_skeleton
        else:
            print('plan is none')
        # write experiment manager output
        if evaluation:
            experiment_manager.set_object_id(-1, 'none.stl')
            if plan_total is None:
                experiment_manager.set_mp_success(False, 1)
            else:
                experiment_manager.set_mp_success(True, 1)
            experiment_manager.set_surface_contact_data(planner.surface_contact_tracker.return_data())
            experiment_data = experiment_manager.get_object_data()
            trial_fname = '%d.npz' % experiment_manager.global_trials
            trial_fname = osp.join(skeleton_data_dir, trial_fname)
            video_trial_fname = '%d.avi' % experiment_manager.global_trials
            video_trial_fname = osp.join(skeleton_data_dir, video_trial_fname)
            np.savez(trial_fname, data=experiment_data)

        if rpo_plan_viz.background_ready and plan_total is not None:
            rpo_plan_viz.render_plan_background(
                plan_skeleton.skeleton_skills, 
                plan_total, 
                scene_pcd,
                video_trial_fname)

        all_plans_processed, all_plans_raw, true_plan_processed, true_plan_raw = planner.process_all_plan_transitions(scene_pcd=scene_pcd)

        if args.trimesh_viz:
            # for plan_total in all_plans_raw:
            if plan_total is not None:
                real_skeleton = []
                for i in range(1, len(plan_total)):
                    real_skeleton.append(plan_total[i].skill)
                skeleton = real_skeleton
                print('skeleton: ', skeleton)
                for ind in range(len(plan_total) - 1):     
                # ind = 0 
                    pcd_data = {}
                    pcd_data['start'] = plan_total[ind].pointcloud_full
                    pcd_data['object_pointcloud'] = plan_total[ind].pointcloud_full
                    pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(plan_total[ind+1].transformation)))
                    pcd_data['contact_world_frame_right'] = np.asarray(plan_total[ind+1].palms[:7])
                    if 'pull' in skeleton[ind] or 'push' in skeleton[ind]:
                        pcd_data['contact_world_frame_left'] = np.asarray(plan_total[ind+1].palms[:7])
                    else:
                        pcd_data['contact_world_frame_left'] = np.asarray(plan_total[ind+1].palms[7:])
                    scene = viz_palms.vis_palms_pcd(pcd_data, world=True, centered=False, corr=False)
                    scene.show()
        
        # from IPython import embed
        # embed()
        log_debug('Adding %d plans to the buffer' % len(all_plans_processed))
        for transition_data in all_plans_processed:
            skeleton_policy.add_to_replay_buffer(rpo_plan2lcm(transition_data))
        if true_plan_processed is not None:
            skeleton_policy.add_to_replay_buffer(rpo_plan2lcm(true_plan_processed))
        trials += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general environment/execution args
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--np_seed', type=int, default=0)
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--trimesh_viz', action='store_true')
    parser.add_argument('--ignore_physics', action='store_true')
    parser.add_argument('-r', '--rolling', type=float, default=0.0, help='rolling friction value for pybullet sim')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--camera_inds', nargs='+', type=int)
    parser.add_argument('--pcd_noise', action='store_true')
    parser.add_argument('--pcd_noise_std', type=float, default=0.0025)
    parser.add_argument('--pcd_noise_rate', type=float, default=0.00025)
    parser.add_argument('--pcd_scalar', type=float, default=0.9)

    # save data args
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--save_rl_data_dir', type=str, default='rl_results')
    parser.add_argument('--exp', type=str, default='debug')

    # configuration 
    parser.add_argument('--task_config_file', type=str, default='default_problems')
    parser.add_argument('--task_problems_path', type=str, default='data/training_tasks')
    parser.add_argument('--full_skeleton_prediction', action='store_true')
    parser.add_argument('--predict_while_planning', action='store_true')

    # point cloud planner args
    parser.add_argument('--max_steps', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--eval_timeout', type=int, default=120)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')

    # other experiment settings
    parser.add_argument('--task_difficulty', type=str, default='medium')

    args = parser.parse_args()
    main(args)
