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
# from replay_buffer import TransitionBuffer


class SkillExplorer(object):
    def __init__(self, skills):
        self.skills = skills

    def sample_skill(self, strategy=None):
        return np.random.choice(self.skills.keys(), 1)


def main(args):
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

    signal.signal(signal.SIGINT, util.signal_handler)
    rospy.init_node('PlayExplore')

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
    # skills['push_right'] = push_right_skill
    # skills['push_left'] = push_left_skill

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

    plans_to_send = []
    planners_to_send = []
    while True:
        # sample a task
        pointcloud, transformation_des, surfaces, task_surfaces = task_sampler.sample('easy')
        pointcloud_sparse = pointcloud[::int(pointcloud.shape[0]/100), :][:100]
        scene_pointcloud = np.concatenate(surfaces.values())

        if args.predict_while_planning:
            predicted_skeleton = None
        else:
            # run the policy to get a skeleton
            predicted_skeleton, predicted_inds = skeleton_policy.predict(pointcloud_sparse, transformation_des)
            log_debug('predicted: ', predicted_skeleton)

        _, predicted_surfaces = separate_skills_and_surfaces(predicted_skeleton)
        # predicted_skills_processed = process_skeleleton_prediction(predicted_skills, skills.keys())

        # predicted_skeleton_processed, predicted_inds = ['pull_right', 'grasp'], [0, 1, 2]

        # table = surfaces[0]  # TODO: cleaner way to handle no table being present in the training samples
        # target_surfaces = [table]*len(predicted_skills_processed)

        target_surfaces = []
        for surface_name in predicted_surfaces:
            surface_pcd = surfaces[surface_name]
            target_surfaces.append(surface_pcd)

        # create unified skeleton object (contains all info about target point clouds and skeleton for planner)
        plan_skeleton = SkillSurfaceSkeleton(
            predicted_skeleton,
            predicted_inds,
            skeleton_surface_pcds=target_surfaces
        )

        # setup planner
        planner = PointCloudTreeLearner(
            start_pcd=pointcloud_sparse,
            start_pcd_full=pointcloud,
            trans_des=transformation_des,
            plan_skeleton=plan_skeleton,
            skills=skills,
            max_steps=args.max_steps,
            skeleton_policy=skeleton_policy,
            motion_planning=False,
            timeout=30
        )

        log_debug('planning!')
        if predicted_skeleton is not None:
            plan = planner.plan()
        else:
            plan = planner.plan_max_length()
        
        if plan is None:
            log_warn('RPO_MP plan not found')
            continue


        # get transition info from the plan that was obtained
        transition_data = planner.process_plan_transitions(plan[1:])

        log_debug('sending data!')
        for _ in range(2):
            # send the data over
            skeleton_policy.add_to_replay_buffer(rpo_plan2lcm(transition_data))

        # train models
        plans_to_send.append(plan)
        planners_to_send.append(planner)
        # if len(plans_to_send) < 5:
        #     continue

        if len(plans_to_send) > 1:
            from IPython import embed
            embed()
            pass
        predicted_inds_to_send = [rpop.skeleton_indices for rpop in planners_to_send]
        # np.savez('test_plans.npz', plans_to_send=plans_to_send)
        # np.savez('test_predicted_inds.npz', predicted_inds_to_send=predicted_inds_to_send)
        # from IPython import embed
        # embed()

        for _ in range(250):
            for i, plan in enumerate(plans_to_send):
                transition_data = planners_to_send[i].process_plan_transitions(plan[1:])
                skeleton_policy.add_to_replay_buffer(rpo_plan2lcm(transition_data))
        
        # log_debug('sent to buffer')
        # embed()
        
        # plan = np.load('test_plan.npz', allow_pickle=True)
        # predicted_inds_to_send = np.load('test_predicted_inds.npz', allow_pickle=True)
        # plan = plan['plan'].tolist()
        # for _ in range(250):
        #     for i, plan in enumerate(plans_to_send):
        #         planner.skeleton_indices = predicted_inds_to_send[i]
        #         planner._make_skill_lang()
        #         transition_data = planner.process_plan_transitions(plan[1:])
        #         skeleton_policy.add_to_replay_buffer(rpo_plan2lcm(transition_data))


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
    parser.add_argument('--save_data_dir', type=str, default='play_transitions')
    parser.add_argument('--exp', type=str, default='debug')

    # configuration 
    parser.add_argument('--task_config_file', type=str, default='default_problems')
    parser.add_argument('--task_problems_path', type=str, default='data/training_tasks')
    parser.add_argument('--full_skeleton_prediction', action='store_true')
    parser.add_argument('--predict_while_planning', action='store_true')

    # point cloud planner args
    parser.add_argument('--max_steps', type=int, default=5)

    args = parser.parse_args()
    main(args)
