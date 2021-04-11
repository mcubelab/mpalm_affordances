import os, os.path as osp
import sys
import time
import numpy as np
from multiprocessing import Pipe, Queue, Manager, Process
import rospkg
import copy

import pybullet as p
from airobot import Robot
from airobot import set_log_level, log_debug, log_info, log_warn, log_critical

from rpo_planning.utils import common as util
from rpo_planning.robot.multicam_env import YumiMulticamPybullet
from rpo_planning.config.multistep_eval_cfg import get_multistep_cfg_defaults
from rpo_planning.execution.motion_playback import OpenLoopMacroActions
from rpo_planning.utils.object import CuboidSampler
from rpo_planning.utils.data import MultiBlockManager
from rpo_planning.utils.motion import guard
from rpo_planning.skills.samplers.pull import PullSamplerBasic, PullSamplerVAE
from rpo_planning.skills.samplers.push import PushSamplerBasic, PushSamplerVAE
from rpo_planning.skills.samplers.grasp import GraspSamplerBasic, GraspSamplerVAE
from rpo_planning.skills.primitive_skills import (
    GraspSkill, PullRightSkill, PullLeftSkill, PushRightSkill, PushLeftSkill
)
from rpo_planning.motion_planning.primitive_planners import (
    grasp_planning_wf, pulling_planning_wf, pushing_planning_wf
)
# from rpo_planning.pointcloud_planning.rpo_learner import PointCloudTreeLearner
from rpo_planning.pointcloud_planning.rpo_planner import PointCloudTree 
from rpo_planning.pointcloud_planning.rpo_learner import PointCloudTreeLearner
from rpo_planning.utils.exceptions import (
    SkillApproachError, InverseKinematicsError, 
    DualArmAlignmentError, PlanWaypointsError, 
    MoveToJointTargetError
)
from rpo_planning.evaluation.exploration.experiment_manager import SimpleRPOEvalManager
from rpo_planning.utils.planning.rpo_plan_visualize import FloatingPalmPlanVisualizer
from rpo_planning.utils.visualize import PalmVis
from rpo_planning.utils.remote.fork_pbd import ForkablePdb


def worker_planner(child_conn, work_queue, result_queue, global_result_queue, global_dict, worker_flag_dict, seed, worker_id):
    while True:
        try:
            if not child_conn.poll(0.0001):
                continue
            msg = child_conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if msg == "INIT":
            args = global_dict['args']
            experiment_cfg = global_dict['experiment_cfg']
            skeleton_data_dir = global_dict['skeleton_data_dir']
            skeleton_exp_name = global_dict['skeleton_exp_name']
            obj_id = goal_obj_id = 100

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

            args.sim = True
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

            prefix_n = worker_id
            skill_names = global_dict['skill_names']

            # prefixes will be used to name the LCM publisher/subscriber channels
            # the NN server will expect the same number of workers, and will name it's pub/sub
            # messages according to the worker_id -- here is where we name the sampler-side
            # interface accordingly
            pull_prefix = 'pull_%d_vae_' % prefix_n
            grasp_prefix = 'grasp_%d_vae_' % prefix_n
            push_prefix = 'push_%d_vae_' % prefix_n
            pull_sampler = PullSamplerVAE(sampler_prefix=pull_prefix)
            push_sampler = PushSamplerVAE(sampler_prefix=push_prefix)
            grasp_sampler = GraspSamplerVAE(sampler_prefix=grasp_prefix, default_target=None)

            avoid_collisions = True 
            ignore_mp = True
            pull_right_skill = PullRightSkill(
                pull_sampler,
                robot=yumi_gs,
                get_plan_func=pulling_planning_wf,
                ignore_mp=ignore_mp,
                avoid_collisions=avoid_collisions
            )

            pull_left_skill = PullLeftSkill(
                pull_sampler,
                robot=yumi_gs,
                get_plan_func=pulling_planning_wf,
                ignore_mp=ignore_mp,
                avoid_collisions=avoid_collisions
            )

            push_right_skill = PushRightSkill(
                push_sampler,
                robot=yumi_gs,
                get_plan_func=pushing_planning_wf,
                ignore_mp=ignore_mp,
                avoid_collisions=avoid_collisions
            )

            push_left_skill = PushLeftSkill(
                push_sampler,
                robot=yumi_gs,
                get_plan_func=pushing_planning_wf,
                ignore_mp=ignore_mp,
                avoid_collisions=avoid_collisions
            )

            grasp_skill = GraspSkill(grasp_sampler, robot=yumi_gs, get_plan_func=grasp_planning_wf, ignore_mp=ignore_mp)
            grasp_pp_skill = GraspSkill(grasp_sampler, robot=yumi_gs, get_plan_func=grasp_planning_wf, pp=True, ignore_mp=ignore_mp)

            skills = {}
            for name in skill_names:
                skills[name] = None
            skills['pull_right'] = pull_right_skill
            skills['pull_left'] = pull_left_skill
            skills['grasp'] = grasp_skill
            skills['grasp_pp'] = grasp_pp_skill            
            skills['push_right'] = push_right_skill
            skills['push_left'] = push_left_skill

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

            # create interface to make guarded movements
            guarder = guard.GuardedMover(robot=yumi_gs, pb_client=yumi_ar.pb_client.get_client_id(), cfg=cfg)       

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

            # set up experiment manager
            experiment_manager = SimpleRPOEvalManager(skeleton_data_dir, skeleton_exp_name, cfg)
            log_info('RPO Planner (eval): Saving results to directory: %s' % skeleton_data_dir)
            print(experiment_cfg)
            
            # prep visualization tools
            palm_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.PALM_MESH_FILE)
            table_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.TABLE_MESH_FILE)
            rpo_plan_viz = FloatingPalmPlanVisualizer(palm_mesh_file, table_mesh_file, cfg, skills)
            viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)

            continue
        if msg == "RESET":
            continue
        if msg == "SAMPLE":        
            worker_flag_dict[worker_id] = False
            # get the task information from the work queue
            # point_cloud, transformation, skeleton
            planner_inputs = work_queue.get()
            pointcloud_sparse = planner_inputs['pointcloud_sparse']
            pointcloud = planner_inputs['pointcloud']
            scene_pointcloud = planner_inputs['scene_pointcloud']
            transformation_des = planner_inputs['transformation_des']
            plan_skeleton = planner_inputs['plan_skeleton']
            max_steps = planner_inputs['max_steps']
            timeout = planner_inputs['timeout']
            eval_timeout = planner_inputs['eval_timeout']
            skillset_cfg = planner_inputs['skillset_cfg']
            surfaces = planner_inputs['surfaces']
            start_pose = planner_inputs['start_pose']
            goal_pose = planner_inputs['goal_pose']
            obj_fname = planner_inputs['fname']
            obj_fname = osp.join(os.environ['CODE_BASE'], 'catkin_ws', obj_fname)

            try:
                yumi_ar.pb_client.remove_body(obj_id)
                yumi_ar.pb_client.remove_body(goal_obj_id)
            except:
                pass

            obj_id, sphere_ids, mesh, goal_obj_id = \
                cuboid_sampler.sample_cuboid_pybullet(
                    obj_fname,
                    goal=True,
                    keypoints=False,
                    scale=[1.0]*3)            
            experiment_manager.set_object_id(obj_id, obj_fname)

            # setup planner
            # TODO: check if we're going to face problems using skeleton_policy = None
            # (which will not be possible if we want to predict high level skills on a per-node basis)
            
            print('Eval planner, skeleton: ', plan_skeleton.skeleton_full)
            planner = PointCloudTreeLearner(
                start_pcd=pointcloud_sparse,
                start_pcd_full=pointcloud,
                trans_des=transformation_des,
                plan_skeleton=plan_skeleton,
                skills=skills,
                max_steps=max_steps,
                skeleton_policy=None,
                skillset_cfg=skillset_cfg,
                pointcloud_surfaces=surfaces,
                motion_planning=True,
                timeout=eval_timeout,
                epsilon=0.0, 
                visualize=True,
                obj_id=goal_obj_id,
                start_pose=util.list2pose_stamped(start_pose)
            )

            p.resetBasePositionAndOrientation(
                obj_id,
                start_pose[:3],
                start_pose[3:])

            log_debug('Planning from RPO_Planning (evaluation) worker ID: %d' % worker_id)
            start_plan_time = time.time()
            # plan_total = planner.plan()
            # skeleton = plan_skeleton.skeleton_full
            plan_total = planner.plan_with_skeleton_explore(evaluate=True)
            planning_time = time.time() - start_plan_time
            skeleton = plan_skeleton.skeleton_full

            result = {} 
            result['worker_id'] = worker_id
            if plan_total is None:
                log_debug('Planning from RPO_Planning (evaluation): Plan not found from worker ID: %d' % worker_id)
                result['data'] = None 
                result['short_plan'] = False 
                experiment_manager.set_mp_success(False, 1)
            else:
                # get transition info from the plan that was obtained
                transition_data = planner.process_plan_transitions(plan_total[1:], scene_pointcloud)
                log_debug('Planning from RPO_Planning (evaluation): Putting transition data for replay buffer into queue from worker ID %d' % worker_id)
                result['data'] = transition_data
                result['short_plan'] = True if len(plan_skeleton.skeleton_full) == 1 else False 
                plan = plan_total[1:]
                experiment_manager.set_mp_success(True, 1)
            
            # write experiment manager output
            experiment_manager.set_surface_contact_data(planner.surface_contact_tracker.return_data())
            experiment_data = experiment_manager.get_object_data()
            trial_fname = '%d.npz' % experiment_manager.global_trials
            trial_fname = osp.join(skeleton_data_dir, trial_fname)
            video_trial_fname = '%d.avi' % experiment_manager.global_trials
            video_trial_fname = osp.join(skeleton_data_dir, video_trial_fname)
            np.savez(trial_fname, data=experiment_data)

            # !!! make sure this queue.put(result) is run! that is how the main process knows to get a new task for the planner
            global_result_queue.put(result)
            worker_flag_dict[worker_id] = True 
            if plan_total is None:
                continue
            # print("HERE!!!!! After we found a successful plan in RPO eval")
            if args.trimesh_viz:
                for ind in range(len(plan_total) - 1):
                # ind = 0 
                    pcd_data = {}
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
            # ForkablePdb().set_trace()
            if rpo_plan_viz.background_ready:
                rpo_plan_viz.render_plan_background(
                    plan_skeleton.skeleton_skills, 
                    plan_total, 
                    scene_pointcloud,
                    video_trial_fname)
            #################### execute plan ################################
            print('RPO_Planning (evaluation): Setting up execution with obj %s' % obj_fname)
            # plan = copy.deepcopy(plan_total[1:])


            # execute plan if one is found...
            pose_plan = [(start_pose, util.list2pose_stamped(start_pose))]
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


            # if args.goal_viz:
            #     goal_viz.update_goal_state(goal_pose)
            #     goal_viz.show_goal_obj()

            real_start_pos = p.getBasePositionAndOrientation(obj_id)[0]
            real_start_ori = p.getBasePositionAndOrientation(obj_id)[1]
            real_start_pose = list(real_start_pos) + list(real_start_ori)
            real_start_mat = util.matrix_from_pose(util.list2pose_stamped(real_start_pose))

            ############# set up trial data ###############
            trial_data = {}
            trial_data['start_pcd'] = pointcloud
            trial_data['start_pcd_down'] = pointcloud_sparse
            trial_data['obj_fname'] = obj_fname 
            trial_data['start_pose'] = np.asarray(real_start_pose)
            trial_data['goal_pose'] = np.asarray(goal_pose)
            trial_data['goal_pose_global'] = np.asarray(goal_pose)
            trial_data['trans_des_global'] = transformation_des

            trial_data['skeleton'] = skeleton

            trial_data['predictions'] = {}
            # model_path1 = grasp_sampler.get_model_path()
            # model_path2 = pull_sampler.get_model_path()
            # model_path3 = push_sampler.get_model_path()
            # trial_data['predictions']['model_path'] = [model_path1, model_path2, model_path3]
            trial_data['predictions']['model_path'] = None
            trial_data['predictions']['skeleton_model_path'] = None

            # save current camera information
            trial_data['camera_inds'] = args.camera_inds
            trial_data['camera_noise'] = None
            if args.pcd_noise:
                trial_data['camera_noise'] = {}
                trial_data['camera_noise']['std'] = args.pcd_noise_std
                trial_data['camera_noise']['rate'] = args.pcd_noise_rate
            
            # get the during planning failure stats
            trial_data['planning_failure'] = planner.planning_stat_tracker.collect_data()

            trial_data['planning_time'] = time.time() - start_plan_time
            goal_obj_id2 = None
            args.playback_num = 1
            args.ignore_physics = True
            guarder.set_object_id(obj_id=obj_id)

            if global_dict['evaluate_execute']:
                try:
                    start_playback_time = time.time()
                    for playback in range(args.playback_num):
                        yumi_ar.arm.go_home(ignore_physics=True)
                        try:
                            yumi_ar.pb_client.remove_body(goal_obj_id2)
                        except:
                            pass
                        # if playback > 0 and goal_visualization:
                        #     goal_viz.hide_goal_obj()
                        # if playback == 2:
                        #     goal_viz.show_goal_obj()
                        #     goal_obj_id2 = yumi_ar.pb_client.load_geom(
                        #         shape_type='mesh',
                        #         visualfile=obj_fname,
                        #         collifile=obj_fname,
                        #         mesh_scale=[1.0]*3,
                        #         base_pos=[0.45, 0, 0.1],
                        #         rgba=[0.0, 0.0, 0.95, 0.25],
                        #         mass=0.03)

                        #     p.setCollisionFilterPair(goal_obj_id2, obj_id, -1, -1, enableCollision=False)
                        #     p.setCollisionFilterPair(goal_obj_id2, goal_obj_id, -1, -1, enableCollision=False)
                        #     cuboid_manager.filter_collisions(obj_id, goal_obj_id2)

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
                                action_planner.add_remove_scene_object(action='add')
                                cuboid_manager.robot_collisions_filter(obj_id, enable=False)
                                time.sleep(0.5)
                                if args.ignore_physics:
                                    yumi_ar.arm.set_jpos(skill_cfg.RIGHT_INIT + skill_cfg.LEFT_INIT,
                                                        ignore_physics=True)
                                else:
                                    _, _ = yumi_gs.move_to_joint_target_mp(skill_cfg.RIGHT_INIT, skill_cfg.LEFT_INIT,
                                                                            execute=True)
                                action_planner.add_remove_scene_object(action='remove')
                                cuboid_manager.robot_collisions_filter(obj_id, enable=True)
                                time.sleep(0.5)

                                # move to making contact, and ensure contact is made
                                _, _ = action_planner.single_arm_setup(full_plan[i][0], pre=True)
                                start_playback_time = time.time()
                                n = True if 'pull' in skill else False
                                if not guarder.still_pulling(arm=arm, n=n):
                                    while True:
                                        if guarder.still_pulling(arm=arm, n=n) or time.time() - start_playback_time > 20.0:
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
                                grasp_success = grasp_success and guarder.still_pulling(n=False)
                                print('grasp success: ' + str(grasp_success))
                                time.sleep(0.5)
                                action_planner.single_arm_retract(arm=arm)

                            elif 'grasp' in skill:
                                action_planner.add_remove_scene_object(action='add')
                                cuboid_manager.robot_collisions_filter(obj_id, enable=False)
                                time.sleep(0.5)
                                if args.ignore_physics:
                                    yumi_ar.arm.set_jpos(grasp_cfg.RIGHT_INIT + grasp_cfg.LEFT_INIT,
                                                        ignore_physics=True)
                                else:
                                    _, _ = yumi_gs.move_to_joint_target_mp(grasp_cfg.RIGHT_INIT, grasp_cfg.LEFT_INIT,
                                                                            execute=True)
                                action_planner.add_remove_scene_object(action='remove')
                                cuboid_manager.robot_collisions_filter(obj_id, enable=True)
                                time.sleep(0.5)

                                _, _ = action_planner.dual_arm_setup(full_plan[i][0], 0, pre=True)
                                action_planner.add_remove_scene_object(action='remove')
                                start_playback_time = time.time()
                                if not guarder.still_grasping():
                                    jj = 0
                                    while True:
                                        if guarder.still_grasping() or time.time() - start_playback_time > 20.0:
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
                                        grasp_success = grasp_success and guarder.still_grasping(n=False)
                                        print('grasp success: ' + str(grasp_success))
                                    time.sleep(1.0)
                    print('RPO_Planning (evaluation): Execution was successful')
                # except (ValueError, TypeError) as e:
                #     print(e)
                except (SkillApproachError, InverseKinematicsError,
                        PlanWaypointsError, DualArmAlignmentError,
                        MoveToJointTargetError) as e:
                    log_warn(e)
                    # experiment_manager.set_mp_success(True, 1)
                    # experiment_manager.set_planning_failure(trial_data['planning_failure'])
                    # experiment_manager.set_execute_success(False)
                    # obj_data = experiment_manager.get_object_data()
                    # if args.save_data:
                    #     print('Saving to: ' + str(obj_data_fname))
                    #     with open(obj_data_fname, 'wb') as f:
                    #         pickle.dump(obj_data, f)

                    # try:
                    #     yumi_ar.pb_client.remove_body(goal_obj_id2)
                    # except:
                    #     pass
                    # continue

                ##################################################################
            yumi_ar.arm.go_home(ignore_physics=True) 
            try:
                yumi_ar.pb_client.remove_body(obj_id)
            except:
                pass
            # execute

            continue
        if msg == "END":
            break        
        time.sleep(0.001)
    log_info('Breaking Worker ID: %d' % worker_id)
    child_conn.close()


class RPOEvalWorkerManager:
    """
    Class to interface with a set of workers running in multiple processes
    using multiprocessing's Pipes, Queues, and Managers. In this case, workers
    are each individual instances of the RPO planner, which takes in a point cloud,
    task specification, and plan skeleton, and returns either a sequence of 
    subgoals and contact poses that reaches the goal or a flag that says the plan
    skeleton is infeasible.

    Attributes:
        global_result_queue (Queue): Queue for the workers to put their results in
        global_manager (Manager): Manager for general purpose shared memory. Used primarily
            to share a global dictionary among the workers
        global_dict (dict): Dictionary with shared global memory among the workers, for
            general-purpose data that should be accessible by all workers
        work_queues (dict): Dictionary keyed by worker id holding Queues for sending worker-specific
            data to the process
        result_queues (dict): Dictionary keyed by worker id holding Queues for receiving worker-specific
            data
        worker_flag_dict (dict): Dictionary with shared global memory among the workers,
            specifically used to flag when workers are ready for a task or have completed a task
    """
    def __init__(self, global_result_queue, global_manager, skill_names, experiment_cfg, num_workers=1):

        self.global_result_queue = global_result_queue
        self.global_manager = global_manager
        self.global_dict = self.global_manager.dict()
        self.global_dict['trial'] = 0
        self.global_dict['skill_names'] = skill_names
        self.global_dict['experiment_cfg'] = experiment_cfg
        self.worker_flag_dict = self.global_manager.dict()        

        self.np_seed_base = 1
        self.setup_workers(num_workers)

    def setup_workers(self, num_workers):
        """Setup function to instantiate the desired number of
        workers. Pipes and Processes set up, stored internally,
        and started.
        Args:
            num_workers (int): Desired number of worker processes
        """
        worker_ids = np.arange(num_workers, dtype=np.int64).tolist()
        seeds = np.arange(self.np_seed_base, self.np_seed_base + num_workers, dtype=np.int64).tolist()

        self._worker_ids = worker_ids
        self.seeds = seeds

        self._pipes = {}
        self._processes = {}
        self.work_queues = {}
        self.result_queues = {}
        for i, worker_id in enumerate(self._worker_ids):
            parent, child = Pipe(duplex=True)
            work_q, result_q = Queue(), Queue()
            self.work_queues[worker_id] = work_q
            self.result_queues[worker_id] = result_q
            self.worker_flag_dict[worker_id] = True
            proc = Process(
                target=worker_planner,
                args=(
                    child,
                    work_q,
                    result_q,
                    self.global_result_queue,
                    self.global_dict,
                    self.worker_flag_dict,
                    seeds[i],
                    worker_id,
                )
            )
            proc.daemon = True
            pipe = {}
            pipe['parent'] = parent
            pipe['child'] = child

            self._pipes[worker_id] = pipe
            self._processes[worker_id] = proc

        for i, worker_id in enumerate(self._worker_ids):
            self._processes[worker_id].start()
            self._pipes[worker_id]['parent'].send('INIT')
            log_debug('RESET WORKER ID: %d' % worker_id)
        log_debug('FINISHED WORKER SETUP')

    def put_worker_work_queue(self, worker_id, data):
        """Setter function for putting things inside the work queue
        for a specific worker

        Args:
            worker_id (int): Worker id to place data inside, for sending to worker
            data (dict): Dictionary with data to put inside of work queue. Data
                should contain information that planner needs to set up it's problem
                TODO: check to see what data fields were included in this data dictionary
        """
        self.work_queues[worker_id].put(data)
        
    def _get_worker_work_queue(self, worker_id):
        """Getter function for the work queue for a specific worker

        Args:
            worker_id (int): Worker id to place data inside, for sending to worker
        """
        return self.work_queues[worker_id].get()
    
    def _put_worker_result_queue(self, worker_id, data):
        """Setter function for putting things inside the result queue
        for a specific worker

        Args:
            worker_id (int): Worker id to place data inside, for sending to worker
            data (TODO): Data to put inside of work queue 
        """
        self.result_queues[worker_id].put(data)
        
    def get_worker_result_queue(self, worker_id):
        """Getter function for the result queue for a specific worker

        Args:
            worker_id (int): Worker id to place data inside, for sending to worker
        """
        return self.result_queues[worker_id].get()
    
    def get_global_result_queue(self):
        """Getter function for the global result queue
        """
        return self.global_result_queue.get()

    def get_global_info_dict(self):
        """Returns the globally shared dictionary of data
        generation information, including success rate and
        trial number

        Returns:
            dict: Dictionary of global information shared
                between workers
        """
        return self.global_dict

    def sample_worker(self, worker_id):
        """Function to send the "sample" command to a specific worker

        Args:
            worker_id (int): ID of which worker we should tell to start running
        """
        self._pipes[worker_id]['parent'].send('SAMPLE')

    def stop_all_workers(self):
        """Function to send an "exit" signal to all workers for clean break
        """
        for worker_id in self._worker_ids:
            self._pipes[worker_id]['parent'].send('END')

    def get_pipes(self):
        return self._pipes

    def get_processes(self):
        return self._processes

    def get_worker_ids(self):
        return self._worker_ids

    def get_worker_ready(self, worker_id):
        return self.worker_flag_dict[worker_id]
