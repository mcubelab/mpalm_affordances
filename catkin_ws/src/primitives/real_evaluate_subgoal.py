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
from macro_actions import ClosedLoopMacroActions, OpenLoopMacroActionsReal

# from closed_loop_experiments_cfg import get_cfg_defaults
from multistep_planning_eval_cfg import get_cfg_defaults
from data_gen_utils import YumiCamsGS, DataManager, YumiCamsGSReal
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

    # use our primitive for global config
    primitive_name = args.primitive_name
    if 'grasp' in primitive_name:
        cfg = grasp_cfg
    elif 'pull' in primitive_name:
        cfg = pull_cfg
    elif 'push' in primitive_name:
        cfg = push_cfg
    else:
        raise ValueError('Primitive name not recognized')

    # create airobot interface
    yumi_ar = Robot('yumi_palms', pb=False)
    yumi_ar.arm.right_arm.set_speed(100, 50)
    yumi_ar.arm.left_arm.set_speed(100, 50)
    yumi_ar.arm.right_arm.start_egm()
    yumi_ar.arm.left_arm.start_egm()

    # create YumiReal interface
    yumi_gs = YumiCamsGSReal(yumi_ar, cfg, n_cam=2)
    print('GOING HOME!')
    time.sleep(1.0)
    _, _ = yumi_gs.move_to_joint_target_mp(yumi_ar.arm.right_arm.cfgs.ARM.HOME_POSITION, yumi_ar.arm.left_arm.cfgs.ARM.HOME_POSITION, execute=True)
    time.sleep(1.0)


    # create action planner
    action_planner = OpenLoopMacroActionsReal(cfg=cfg, robot=yumi_gs, pb=False) # TODO

    ### prepare for planner structures: filesystem io for predictions, samplers, skills

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
        grasp_sampler = GraspSamplerVAEPubSub(default_target=None, obs_dir=obs_dir, pred_dir=pred_dir)
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
    viz_pcd = PCDVis()

    # get it going
    while not rospy.is_shutdown():
        # get details of planning problem
        if args.demo:
            pass
        else:
            # if args.skeleton == 'pgp':
                # get desired transformation # TODO
            pass

        print('Getting new observation')
        from IPython import embed
        embed()
        time.sleep(1.0)
        _, _ = yumi_gs.move_to_joint_target_mp(yumi_ar.arm.right_arm.cfgs.ARM.HOME_POSITION, yumi_ar.arm.left_arm.cfgs.ARM.HOME_POSITION, execute=True)
        time.sleep(5.0)        

        # obtain observation from sensors!
        # TODO!
        obs, pcd = yumi_gs.get_observation()

        # preprocess the observation for providing to the NN
        pointcloud_pts = np.asarray(obs['down_pcd_pts'][:100, :], dtype=np.float32)
        pointcloud_pts_full = np.asarray(np.concatenate(obs['pcd_pts']), dtype=np.float32)
        table_pts = np.concatenate(obs['table_pcd_pts'], axis=0)[::500, :]

        # update target pointcloud for grasping, based on where the table is
        # grasp_sampler.update_default_target(np.concatenate(obs['table_pcd_pts'], axis=0)[::500, :])

        # sample from the model
        start_state = PointCloudNode()
        start_state.set_pointcloud(
            pcd=pointcloud_pts,
            pcd_full=pointcloud_pts_full
        )
        target_surface = table_pts  # TODO

        # sample an action
        new_state = skills[primitive_name].sample(
                    start_state,
                    target_surface=target_surface,
                    final_trans=False
                )            

        # # post-process the prediction into a new point cloud node
        # new_state = PointCloudNode()
        # new_state.init_state(start_state, prediction['transformation'])
        # correction = True

        # dual = True
        # if primitive_name != 'grasp':
        #     dual = False

        # new_state.init_palms(prediction['palms'],
        #                      correction=correction,
        #                      prev_pointcloud=start_state.pointcloud_full,
        #                      dual=dual)

        # if 'pull' in primitive_name:
        #     new_state.palms[2] -= 0.0035

        if args.trimesh_viz:
            viz_data = {}
            viz_data['contact_world_frame_right'] = new_state.palms_raw[:7]
            if 'grasp' in primitive_name:
                viz_data['contact_world_frame_left'] = new_state.palms_raw[7:]
            else:
                viz_data['contact_world_frame_left'] = new_state.palms_raw[:7]
            viz_data['transformation'] = util.pose_stamped2np(util.pose_from_matrix(new_state.transformation))
            viz_data['object_pointcloud'] = pointcloud_pts_full
            viz_data['start'] = pointcloud_pts_full

            scene_pcd = vis_palms.vis_palms_pcd(viz_data, world=True, corr=False, full_path=True, show_mask=False, goal_number=1)
            scene_pcd.show()

        # get open loop plan from primitive planners
        trans_execute = util.pose_from_matrix(new_state.transformation)

        # try to execute the action
        if 'grasp' in primitive_name:
            local_plan = grasp_planning_wf(
                util.list2pose_stamped(new_state.palms[7:]),
                util.list2pose_stamped(new_state.palms[:7]),
                trans_execute
            )
        elif 'pull' in primitive_name:
            local_plan = pulling_planning_wf(
                util.list2pose_stamped(new_state.palms[:7]),
                util.list2pose_stamped(new_state.palms[:7]),
                trans_execute,
                N=150
            )
        elif 'push' in primitive_name:
            local_plan = pushing_planning_wf(
                util.list2pose_stamped(new_state.palms[:7]),
                util.list2pose_stamped(new_state.palms[:7]),
                trans_execute,
                arm='r',
                G_xy=(new_state.palms[:2] - np.mean(start_state.pointcloud_full, axis=0)[:-1])
            )

        # TODO
        if args.rviz_viz:
            from IPython import embed
            embed()            
        #     import simulation
        #     for i in range(10):
        #         simulation.visualize_object(
        #             start_pose,
        #             filepath="package://config/descriptions/meshes/objects/cuboids/" +
        #                 cuboid_fname.split('objects/cuboids')[1],
        #             name="/object_initial",
        #             color=(1., 0., 0., 1.),
        #             frame_id="/yumi_body",
        #             scale=(1., 1., 1.))
        #         simulation.visualize_object(
        #             des_goal_pose,
        #             filepath="package://config/descriptions/meshes/objects/cuboids/" +
        #                 cuboid_fname.split('objects/cuboids')[1],
        #             name="/object_final",
        #             color=(0., 0., 1., 1.),
        #             frame_id="/yumi_body",
        #             scale=(1., 1., 1.))
        #         rospy.sleep(.1)
        #     simulation.simulate_palms(local_plan, cuboid_fname.split('objects/cuboids')[1])

        # full motion planning check to see if the plan is feasible
        feasible = skills[primitive_name].feasible_motion(
            state=new_state,
            start_joints=None,
            nominal_plan=local_plan)

        if not feasible:
            print('not feasible')
            continue        

        print('Ready for motion')
        from IPython import embed
        embed()

        if 'grasp' in primitive_name:
            # try to execute the action
            # yumi_ar.arm.set_jpos(grasp_cfg.RIGHT_INIT + grasp_cfg.LEFT_INIT)
            # action_planner.add_remove_scene_object('add')
            # time.sleep(0.5)
            _, _ = yumi_gs.move_to_joint_target_mp(grasp_cfg.RIGHT_INIT, grasp_cfg.LEFT_INIT, execute=True)
            time.sleep(9.0)
            # action_planner.add_remove_scene_object('remove')
            # time.sleep(0.5)
            try:
                for k, subplan in enumerate(local_plan):
                    time.sleep(1.0)
                    action_planner.playback_dual_arm('grasp', subplan, k)
            except ValueError as e:
            	print(e)
                continue
        elif 'pull' in primitive_name or 'push' in primitive_name:
            print('Executing!')
            skill_cfg = pull_cfg if 'pull' in primitive_name else push_cfg
            # set arm configuration to good start state
            # action_planner.add_remove_scene_object('add')
            # time.sleep(0.5)
            _, _ = yumi_gs.move_to_joint_target_mp(skill_cfg.RIGHT_INIT, skill_cfg.LEFT_INIT, execute=True)
            time.sleep(5.0)
            # action_planner.add_remove_scene_object('remove')                                
            # time.sleep(0.5)        	
            try:
                action_planner.playback_single_arm(primitive_name, local_plan[0])
            except ValueError as e:
                print(e)
                continue


        from IPython import embed
        embed()
        for _ in range(5):
            action_planner.single_arm_retract()
            time.sleep(0.05)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


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
        '--example_config_path',
        type=str,
        default='catkin_ws/src/primitives/config')

    parser.add_argument(
        '--primitive_name',
        type=str,
        default='pull_right',
        help='which primitive to plan')


    parser.add_argument(
        '--debug', action='store_true'
    )


    parser.add_argument(
        '--trimesh_viz', action='store_true'
    )

    parser.add_argument(
        '--rviz_viz', action='store_true'
    )    

    parser.add_argument(
        '--skeleton', type=str, default='pg'
    )

    parser.add_argument(
        '--demo', action='store_true')

    parser.add_argument(
        '--baseline', action='store_true')

    args = parser.parse_args()
    main(args)