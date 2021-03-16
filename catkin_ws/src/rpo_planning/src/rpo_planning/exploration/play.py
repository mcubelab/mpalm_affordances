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
from rpo_planning.robot.multicam_env import YumiMulticamPybullet 
from rpo_planning.utils.object import CuboidSampler
from rpo_planning.utils.pb_visualize import GoalVisual
from rpo_planning.utils.data import MultiBlockManager
from rpo_planning.utils.motion import guard
from rpo_planning.utils.planning.pointcloud_plan import PointCloudNode
from rpo_planning.utils.visualize import PCDVis, PalmVis
from rpo_planning.utils.exceptions import (
    SkillApproachError, InverseKinematicsError, 
    DualArmAlignmentError, PlanWaypointsError, 
    MoveToJointTargetError
)

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


class SkillExplorer(object):
    def __init__(self, skills):
        self.skills = skills

    def sample_skill(self, strategy=None):
        # return np.random.choice(self.skills.keys(), 1)
        return random.sample(self.skills, 1)[0]


def main(args):
    # example_config_path = osp.join(os.environ['CODE_BASE'], args.example_config_path)
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
    # skills['pull_right'] = pull_right_skill
    # skills['pull_left'] = pull_left_skill
    # skills['grasp'] = grasp_skill
    # skills['grasp_pp'] = grasp_pp_skill
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

    # create interface to make guarded movements
    guarder = guard.GuardedMover(robot=yumi_gs, pb_client=yumi_ar.pb_client.get_client_id(), cfg=cfg)

    # create manager
    env = PlayEnvironment(pb_client=yumi_ar.pb_client, robot=yumi_gs, cuboid_manager=cuboid_manager, cuboid_sampler=cuboid_sampler, target_surface_ids=[cfg.TABLE_ID])

    # sample some objects
    env.sample_objects(n=1)

    # instantiate objects in the world at specific poses
    # env.initialize_object_states()

    # setup save stuff
    trasition_save_dir = osp.join(
        os.environ['CODE_BASE'], 
        'catkin_ws/src/primitives', 
        args.data_dir, args.save_data_dir, args.exp)
    if not osp.exists(trasition_save_dir):
        os.makedirs(trasition_save_dir)
    total_transitions = 0

    time.sleep(1.0)
    while True:
        yumi_ar.arm.go_home(ignore_physics=True)

        if not env.check_environment_status():
            env.initialize_object_states()
        # should we wait until everything in the environment is stable?
        # TODO: check if objects are stable
        # in the meantime, just use a sleep to wait long enough
        time.sleep(1.0)

        feasible = False
        current_obs_trial = 0
        # sample a skill type
        skill_type = agent.sample_skill()
        log_debug('Sampling with skill: %s' % skill_type)
        while not feasible:
            if current_obs_trial > 5:
                break
            current_obs_trial += 1
            # get an observation from the environment
            log_debug('Getting observation')
            obs = env.get_observation()
            obj_info = env.get_current_obj_info()[0]
            if args.sim:
                action_planner.update_object(obj_id=obj_info['obj_id'], mesh_file=obj_info['fname'])
                guarder.set_object_id(obj_id=obj_info['obj_id'])
                p.changeDynamics(
                    obj_info['obj_id'],
                    -1,
                    lateralFriction=1.0
                )

            # sample an object from list of object point clouds
            ind = np.random.randint(len(obs['object_pcds']))
            obj_pcd = obs['object_pcds'][ind]

            # sample a target surface
            target_ind = np.random.randint(len(obs['surface_pcds']))
            target_surface = obs['surface_pcds'][target_ind]

            pointcloud_pts = obj_pcd[::int(obj_pcd.shape[0]/100.0)][:100]
            pointcloud_pts_full = obj_pcd

            # scale down point cloud to ensure some contact is made
            centroid = np.mean(pointcloud_pts_full, axis=0)
            centered_pts = pointcloud_pts - centroid
            centered_pts_full = pointcloud_pts_full - centroid
            centered_pts *= args.pcd_scalar 
            centered_pts_full *= args.pcd_scalar
            pointcloud_pts = centered_pts + centroid
            pointcloud_pts_full = centered_pts_full + centroid

            # construct input to agent
            start_sample = PointCloudNode()
            start_sample.set_pointcloud(
                pcd=pointcloud_pts,
                pcd_full=pointcloud_pts_full
            )

            # planes = pcd_segmenter.get_pointcloud_planes(pointcloud_pts_full)
            # start_sample.set_planes(planes)

            # sample an action
            log_debug('Sampling from skill')
            new_state = agent.skills[skill_type].sample(
                        start_sample,
                        target_surface=target_surface,
                        final_trans=False
                    )

            primitive_name = skill_type
            trans_execute = util.pose_from_matrix(new_state.transformation)

            if args.trimesh_viz:
                pcd_data = {}
                pcd_data['start'] = start_sample.pointcloud_full
                pcd_data['object_pointcloud'] = start_sample.pointcloud_full
                pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(new_state.transformation)))
                pcd_data['contact_world_frame_right'] = np.asarray(new_state.palms[:7])
                if 'pull' in skill_type or 'push' in skill_type:
                    pcd_data['contact_world_frame_left'] = np.asarray(new_state.palms[:7])
                else:
                    pcd_data['contact_world_frame_left'] = np.asarray(new_state.palms[7:])
                scene = viz_palms.vis_palms_pcd(pcd_data, world=True, centered=False, corr=False)
                scene.show()

            # try to execute the action
            if 'grasp' in skill_type:
                local_plan = grasp_planning_wf(
                    util.list2pose_stamped(new_state.palms[7:]),
                    util.list2pose_stamped(new_state.palms[:7]),
                    trans_execute
                )
            elif 'pull' in skill_type:
                local_plan = pulling_planning_wf(
                    util.list2pose_stamped(new_state.palms[:7]),
                    util.list2pose_stamped(new_state.palms[:7]),
                    trans_execute
                )
            elif 'push' in skill_type:
                local_plan = pushing_planning_wf(
                    util.list2pose_stamped(new_state.palms[:7]),
                    util.list2pose_stamped(new_state.palms[:7]),
                    trans_execute,
                    arm='r',
                    G_xy=(new_state.palms[:2] - np.mean(start_sample.pointcloud_full, axis=0)[:-1])
                )

            # get any objects that are left behind in the planning scene out of there
            action_planner.clear_planning_scene()

            # full motion planning check to see if the plan is feasible
            log_debug('Checking feasibility')
            feasible = agent.skills[skill_type].feasible_motion(
                state=new_state,
                start_joints=None,
                nominal_plan=local_plan)

        if not feasible:
            log_debug('Not feasible...')
            continue
        active_arm = 'right' if 'right' in skill_type else 'left'
        action_planner.active_arm = active_arm

        try:
            if args.sim:
                if 'push' in skill_type:
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

            if 'grasp' in skill_type:
                # collision free motion to good start configuration
                if args.sim:
                    action_planner.add_remove_scene_object(action='add')
                    time.sleep(0.5)
                    _, _ = yumi_gs.move_to_joint_target_mp(grasp_cfg.RIGHT_INIT, grasp_cfg.LEFT_INIT, execute=True)
                    action_planner.add_remove_scene_object(action='remove')
                    time.sleep(0.5)

                # begin guarded move
                log_debug('Performing dual arm setup')
                _, _ = action_planner.dual_arm_setup(local_plan[0], 0, pre=True)
                start_playback_time = time.time()
                if not guarder.still_grasping():
                    jj = 0
                    while True:
                        if guarder.still_grasping() or time.time() - start_playback_time > 20.0:
                            jj += 1
                        if jj > 2:
                            break
                        log_debug('Performing dual arm approach')
                        # TODO: deal with how this IK approach can break things
                        action_planner.dual_arm_approach()
                        time.sleep(0.075)
                        new_plan = grasp_planning_wf(
                            yumi_gs.get_current_tip_poses()['left'],
                            yumi_gs.get_current_tip_poses()['right'],
                            trans_execute
                        )
                    playback_plan = new_plan
                else:
                    playback_plan = local_plan
                log_debug('Performing dual arm playback')
                for k, subplan in enumerate(playback_plan):
                    action_planner.playback_dual_arm('grasp', subplan, k, pre=False)
                    time.sleep(1.0)
            elif 'pull' in skill_type or 'push' in skill_type:
                skill_cfg = pull_cfg if 'pull' in skill_type else push_cfg
                # start at a good configuration
                if args.sim:
                    action_planner.add_remove_scene_object(action='add')
                    time.sleep(0.5)
                    _, _ = yumi_gs.move_to_joint_target_mp(skill_cfg.RIGHT_INIT, skill_cfg.LEFT_INIT, execute=True)
                    action_planner.add_remove_scene_object(action='remove')
                    time.sleep(0.5)

                
                # perform guarded move to make contact
                _, _ = action_planner.single_arm_setup(local_plan[0], pre=True)
                start_playback_time = time.time()
                n = True if 'pull' in skill_type else False
                if not guarder.still_pulling(arm=active_arm, n=n):
                    while True:
                        if guarder.still_pulling(arm=active_arm, n=n) or time.time() - start_playback_time > 20.0:
                            break
                        action_planner.single_arm_approach(arm=active_arm)
                        time.sleep(0.075)
                        replan_args = [
                            yumi_gs.get_current_tip_poses()['left'],
                            yumi_gs.get_current_tip_poses()['right'],
                            trans_execute
                        ]
                        if 'pull' in skill_type:
                            new_plan = pulling_planning_wf(*replan_args)
                        else:
                            new_plan = pushing_planning_wf(*replan_args)

                    playback_plan = new_plan
                else:
                    playback_plan = local_plan

                # execute open loop motion
                action_planner.playback_single_arm(skill_type, playback_plan[0], pre=False)
                action_planner.single_arm_retract(active_arm)
        except (SkillApproachError, InverseKinematicsError,
                PlanWaypointsError, DualArmAlignmentError,
                MoveToJointTargetError) as e:
            log_warn(e)
            continue

        time.sleep(1.0)
        yumi_ar.arm.go_home(ignore_physics=True)
        time.sleep(1.0)

        # check if environment state is still valid
        if env.check_environment_status():
        # if np.random.random() > 0.5:
            # get new obs
            obs_new = env.get_observation()

            o = obj_pcd
            To = np.asarray(util.pose_stamped2list(util.pose_from_matrix(new_state.transformation)))
            Tpc = np.asarray(new_state.palms)
            o_next = obs_new['object_pcds'][ind]

            # store the transition
            total_transitions += 1
            transition_fname = osp.join(trasition_save_dir, '%d.npz' % total_transitions)
            info_string = 'Saving transition number %d using action %s to fname: %s' % (total_transitions, skill_type, transition_fname)
            print(info_string)
            np.savez(transition_fname,
                observation=o,
                action_type=skill_type,
                contact=Tpc,
                subgoal=To,
                next_observation=o_next
            )
            # transition = (o, skill_type, To, Tpc, o_next)

        else:
            # reset if we're not valid
            env.sample_objects(n=1)
            env.initialize_object_states()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--save_data_dir', type=str, default='play_transitions')
    parser.add_argument('--exp', type=str, default='debug')
    parser.add_argument('--config_package_path', type=str, default='catkin_ws/src/config/')
    parser.add_argument('--example_config_path', type=str, default='catkin_ws/src/primitives/config')
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--np_seed', type=int, default=1000)
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
    parser.add_argument('--debug', action='store_true') 

    args = parser.parse_args()
    main(args)
