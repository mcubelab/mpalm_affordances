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

from airobot import Robot
from airobot.utils import common

sys.path.append(osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives'))

from helper import util
from macro_actions import OpenLoopMacroActions

from multistep_planning_eval_cfg import get_cfg_defaults
from data_gen_utils import YumiCamsGSReal
import simulation
from helper import registration as reg
from helper.pointcloud_planning import PointCloudTree
from helper.pointcloud_planning_utils import PointCloudNode
from helper.pull_samplers import PullSamplerBasic, PullSamplerVAEPubSub
from helper.grasp_samplers import GraspSamplerVAEPubSub, GraspSamplerBasic
from helper.push_samplers import PushSamplerVAEPubSub
from helper.skills import GraspSkill, PullRightSkill, PullLeftSkill, PushRightSkill, PushLeftSkill

from planning import grasp_planning_wf, pulling_planning_wf, pushing_planning_wf
from eval_utils.visualization_tools import PCDVis, PalmVis
from eval_utils.experiment_recorder import GraspEvalManager

def signal_handler(sig, frame):
    """Capture exit signal from keyboard

    Args:
        sig ([type]): [description]
        frame ([type]): [description]
    """
    print('Exit')
    sys.exit(0)


def main(args):
    example_config_path = osp.join(os.environ['CODE_BASE'], args.example_config_path)
    # get cfgs for each primitive
    pull_cfg_file = osp.join(example_config_path, 'pull') + '.yaml'
    pull_cfg = get_cfg_defaults()
    pull_cfg.merge_from_file(pull_cfg_file)
    pull_cfg.freeze()

    grasp_cfg_file = osp.join(example_config_path, 'grasp') + '.yaml'
    grasp_cfg = get_cfg_defaults()
    grasp_cfg.merge_from_file(grasp_cfg_file)
    grasp_cfg.freeze()

    push_cfg_file = osp.join(example_config_path, 'push') + '.yaml'
    push_cfg = get_cfg_defaults()
    push_cfg.merge_from_file(push_cfg_file)
    push_cfg.freeze()

    # for now use pull as main cfg
    cfg = pull_cfg

    # create airobot interface
    yumi_ar = Robot('yumi_palms', pb=False)

    # initialize robot params
    yumi_ar.arm.right_arm.set_speed(100, 50)
    yumi_ar.arm.left_arm.set_speed(100, 50)
    yumi_ar.arm.right_arm.start_egm()
    yumi_ar.arm.left_arm.start_egm()

    # create YumiReal interface
    yumi_gs = YumiCamsGSReal(yumi_ar, cfg, n_cam=2)
    time.sleep(1.0)
    _, _ = yumi_gs.move_to_joint_target_mp(yumi_ar.arm.right_arm.cfgs.ARM.HOME_POSITION, yumi_ar.arm.left_arm.cfgs.ARM.HOME_POSITION, execute=True)
    time.sleep(1.0)

    # create action planner
    action_planner = OpenLoopMacroActions(cfg=cfg, robot=yumi_gs, pb=False)

    # directories used internally for hacky Python 2 to Python 3 pub/sub (get NN predictions using filesystem)
    pred_dir = cfg.PREDICTION_DIR
    obs_dir = cfg.OBSERVATION_DIR
    if not osp.exists(pred_dir):
        os.makedirs(pred_dir)
    if not osp.exists(obs_dir):
        os.makedirs(obs_dir)

    # setup samplers
    if args.baseline:
        print('LOADING BASELINE SAMPLERS')
        pull_sampler = PullSamplerBasic()
        grasp_sampler = GraspSamplerBasic(None)
        push_sampler = PushSamplerVAEPubSub(obs_dir=obs_dir, pred_dir=pred_dir)
    else:
        print('LOADING LEARNED SAMPLERS')
        pull_sampler = PullSamplerVAEPubSub(obs_dir=obs_dir, pred_dir=pred_dir)
        grasp_sampler = GraspSamplerVAEPubSub(default_target=None,
                                              obs_dir=obs_dir, pred_dir=pred_dir)
        push_sampler = PushSamplerVAEPubSub(obs_dir=obs_dir, pred_dir=pred_dir)

    # setup skills
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

    grasp_skill = GraspSkill(
        grasp_sampler,
        yumi_gs,
        grasp_planning_wf
    )

    grasp_pp_skill = GraspSkill(
        grasp_sampler,
        yumi_gs,
        grasp_planning_wf,
        pp=True
    )

    skills = {}
    skills['pull_right'] = pull_right_skill
    skills['pull_left'] = pull_left_skill
    skills['grasp'] = grasp_skill
    skills['grasp_pp'] = grasp_pp_skill
    skills['push_right'] = push_right_skill
    skills['push_left'] = push_left_skill

    # empty the directories so nothing is messed up from previous runs
    pred_fnames, obs_fnames = os.listdir(pred_dir), os.listdir(obs_dir)
    if len(pred_fnames) > 0:
        for fname in pred_fnames:
            os.remove(osp.join(pred_dir, fname))
    if len(obs_fnames) > 0:
        for fname in obs_fnames:
            os.remove(osp.join(obs_dir, fname))

    # prepare for visualization tools
    palm_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.PALM_MESH_FILE)
    table_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.TABLE_MESH_FILE)
    vis_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)

    # get it going
    while not rospy.is_shutdown():
        ### setup task ###

        # make sure robot is at home
        time.sleep(1.0)
        _, _ = yumi_gs.move_to_joint_target_mp(yumi_ar.arm.right_arm.cfgs.ARM.HOME_POSITION, yumi_ar.arm.left_arm.cfgs.ARM.HOME_POSITION, execute=True)
        time.sleep(5.0)

        # get goal pose # TODO
        goal_pose = util.unit_pose()
        # get start pose # TODO
        start_pose = util.unit_pose()
        # compute transformation_des
        transformation_des = util.get_transform(goal_pose, start_pose)

        # get skeleton
        skeleton = raw_input('Please input desired plan skeleton, from options ["pg", "gp", "pgp"]')
        assert skeleton in ['pg', 'gp', 'pgp'], 'Skeleton not recognized, exiting'

        # setup skeleton
        target_surface_skeleton = None
        if skeleton == 'pg':
            skeleton = ['pull_right', 'grasp']
        elif skeleton == 'gp':
            skeleton = ['grasp', 'pull_right']
        elif skeleton == 'pgp':
            skeleton = ['pull_right', 'grasp', 'pull_right']
        else:
            raise ValueError('Unrecognized plan skeleton')

        # get observation
        obs, pcd = yumi_gs.get_observation()

        # process observation
        pointcloud_pts = np.asarray(obs['down_pcd_pts'][:100, :], dtype=np.float32)
        pointcloud_pts_full = np.asarray(np.concatenate(obs['pcd_pts']), dtype=np.float32)
        table_pts = np.concatenate(obs['table_pcd_pts'], axis=0)[::125, :]

        # only use table pointcloud during grasp sampling at the moment
        grasp_sampler.update_default_target(table_pts)

        # setup planner
        planner = PointCloudTree(
            start_pcd=pointcloud_pts,
            trans_des=transformation_des,
            skeleton=skeleton,
            skills=skills,
            max_steps=args.max_steps,
            start_pcd_full=pointcloud_pts_full,
            target_surfaces=target_surface_skeleton
        )

        raw_input('***Ready to plan, press enter to begin planning***')
        # plan, returns None if no plan found before timeout
        if args.no_skeleton:
            plan_total = planner.plan_max_length()
            skeleton = []
            if plan_total is not None:
                for node in plan_total:
                    if node.skill is not None:
                        skeleton.append(node.skill)
        else:
            plan_total = planner.plan()

        if plan_total is None:
            print('Could not find plan')
            continue

        if args.trimesh_viz:
            ind = 2
            pcd_data = {}
            pcd_data['start'] = plan_total[ind].pointcloud_full
            pcd_data['object_pointcloud'] = plan_total[ind].pointcloud_full
            pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(plan_total[ind+1].transformation)))
            pcd_data['contact_world_frame_right'] = np.asarray(plan_total[ind+1].palms[:7])
            if 'pull' in skeleton[ind]:
                pcd_data['contact_world_frame_left'] = np.asarray(plan_total[ind+1].palms[:7])
            else:
                pcd_data['contact_world_frame_left'] = np.asarray(plan_total[ind+1].palms[7:])
            scene = vis_palms.vis_palms_pcd(pcd_data, world=True, centered=False, corr=False)
            scene.show()

        plan = copy.deepcopy(plan_total[1:])

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

        if args.rviz_viz:
            for ii, skill in enumerate(skeleton):
                local_plan = full_plan[ii]
                for i in range(10):
                    simulation.visualize_object(
                        start_pose,
                        filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
                        name="/object_initial",
                        color=(1., 0., 0., 1.),
                        frame_id="/yumi_body",
                        scale=(1., 1., 1.))
                    simulation.visualize_object(
                        goal_pose,
                        filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
                        name="/object_final",
                        color=(0., 0., 1., 1.),
                        frame_id="/yumi_body",
                        scale=(1., 1., 1.))
                    rospy.sleep(.1)
                simulation.simulate_palms(local_plan)

        raw_input('Ready for execution, press enter to begin plan playback')
        try:
            for i, skill in enumerate(skeleton):
                if 'left' in skill:
                    arm = 'left'
                    action_planner.active_arm = 'left'
                    action_planner.inactive_arm = 'right'
                else:
                    arm = 'right'
                    action_planner.active_arm = 'right'
                    action_planner.inactive_arm = 'left'

                if 'pull' in skill or 'push' in skill:
                    skill_cfg = pull_cfg if 'pull' in skill else push_cfg
                    _, _ = yumi_gs.move_to_joint_target_mp(skill_cfg.RIGHT_INIT, cfg.LEFT_INIT, execute=True)
                    time.sleep(5.0)
                    action_planner.playback_single_arm(skill, full_plan[i][0])
                    time.sleep(0.5)
                    action_planner.single_arm_retract(arm=arm, repeat=3)
                elif 'grasp' in skill:
                    _, _ = yumi_gs.move_to_joint_target_mp(grasp_cfg.RIGHT_INIT, grasp_cfg.LEFT_INIT, execute=True)
                    time.sleep(9.0)
                    for k, subplan in enumerate(full_plan[i]):
                        time.sleep(1.0)
                        action_planner.playback_dual_arm('grasp', subplan, k)
                        time.sleep(1.0)
                    action_planner.dual_arm_retract(repeat=2)
        except ValueError as e:
            print(e)
            continue
