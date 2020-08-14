import os
import sys
import time
import argparse
import numpy as np
import rospy
import signal
import threading
import trimesh
from IPython import embed

from airobot import Robot
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from airobot.utils import common
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions
from yumi_pybullet_ros import YumiGelslimPybullet
from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

from closed_loop_experiments_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler
from primitive_data_gen import MultiBlockManager#, GoalVisual
from data_gen_utils import GoalVisual
import simulation


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


class MultiStepManager(object):
    def __init__(self, cfg, grasp_cfg, pull_cfg, push_cfg=None):
        self.grasp_cfg = grasp_cfg
        self.pull_cfg = pull_cfg
        self.push_cfg = push_cfg
        self.cfg = cfg

        # mappings between the stable orientation indices for grasping/pulling
        self.grasp2pull = cfg.GRASP_TO_PULL
        self.pull2grasp = cfg.PULL_TO_GRASP

    def get_args(self, exp_single, exp_double, goal_face, 
                 start_face=None, execute=False, skeleton=['pull', 'grasp', 'pull']):
        """
        Get the primitive arguments for a multi step plan
        """
        if start_face is None:
            start_face = exp_double.get_valid_ind()

        grasp_args = exp_double.get_random_primitive_args(
            ind=start_face,
            random_goal=True,
            execute=execute)
        pull_args_start = exp_single.get_random_primitive_args(
            ind=start_face,
            random_goal=True,
            execute=execute,
            start_pose=grasp_args['object_pose1_world'],
            primitive=skeleton[0])
        pull_args_goal = exp_single.get_random_primitive_args(
            ind=goal_face,
            random_goal=True,
            execute=execute,
            start_pose=grasp_args['object_pose2_world'],
            primitive=skeleton[-1])        

        tmp = pull_args_start['object_pose1_world']
        pull_args_start['object_pose1_world'] = pull_args_start['object_pose2_world']
        pull_args_start['object_pose2_world'] = tmp        

        full_args = [pull_args_start, grasp_args, pull_args_goal]
        return full_args[1:]


def main(args):
    pull_cfg_file = os.path.join(args.example_config_path, 'pull') + ".yaml"
    pull_cfg = get_cfg_defaults()
    pull_cfg.merge_from_file(pull_cfg_file)
    pull_cfg.freeze()

    grasp_cfg_file = os.path.join(args.example_config_path, 'grasp') + ".yaml"
    grasp_cfg = get_cfg_defaults()
    grasp_cfg.merge_from_file(grasp_cfg_file)
    grasp_cfg.freeze()

    push_cfg_file = os.path.join(args.example_config_path, 'push') + ".yaml"
    push_cfg = get_cfg_defaults()
    push_cfg.merge_from_file(push_cfg_file)
    push_cfg.freeze()    

    # cfg = grasp_cfg
    cfg = pull_cfg

    rospy.init_node('MultiStep')
    signal.signal(signal.SIGINT, signal_handler)

    data_seed = args.np_seed
    np.random.seed(data_seed)

    yumi_ar = Robot('yumi_palms',
                    pb=True,
                    pb_cfg={'gui': args.visualize},
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

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        l_gel_id,
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

    yumi_gs = YumiGelslimPybullet(
        yumi_ar,
        cfg,
        exec_thread=False
    )

    for _ in range(10):
        yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    # stl_file = os.path.join(args.config_package_path, 'descriptions/meshes/objects/mustard_1k.stl')
    # stl_file_pb = os.path.join(args.config_package_path, 'descriptions/meshes/objects/mustard_centered.stl')
    
    # obj_name = 'test_cuboid_smaller_'+str(np.random.randint(4999))
    # obj_name = 'test_cuboid_smaller_2711'
    # stl_file = os.path.join(args.config_package_path, 'descriptions/meshes/objects/cuboids', obj_name+'.stl')
    # stl_file_pb = stl_file

    # obj_name = 'test_cylinder_'+str(np.random.randint(4999))
    
    # obj_name = 'test_cylinder_1004'
    # obj_name = 'test_cylinder_104'
    obj_name = 'test_cylinder_1928'
    

    stl_file = os.path.join(args.config_package_path, 'descriptions/meshes/objects/cylinders', obj_name + '.stl')
    # stl_file = os.path.join(args.config_package_path, 'descriptions/meshes/objects/cylinder_simplify_60.stl')
    stl_file_pb = stl_file
    # stl_file = os.path.join(args.config_package_path, 'descriptions/meshes/objects/ycb_objects', args.object_name+'.stl')
    
    # rgba=[0.7, 0.2, 0.2, 1.0]
    rgba_tro=[0.118, 0.463, 0.6, 1.0]
    
    tmesh = trimesh.load_mesh(stl_file)
    init_pose = tmesh.compute_stable_poses()[0][0]
    pos = init_pose[:-1, -1]
    ori = common.rot2quat(init_pose[:-1, :-1])
    obj_id = yumi_ar.pb_client.load_geom(
        shape_type='mesh',
        visualfile=stl_file_pb,
        collifile=stl_file_pb,
        mesh_scale=[1.0, 1.0, 1.0],
        base_pos=[0.45, 0, pos[-1]],
        base_ori=ori,
        rgba=rgba_tro,
        mass=0.03)

    goal_obj_id = yumi_ar.pb_client.load_geom(
        shape_type='mesh',
        visualfile=stl_file_pb,
        collifile=stl_file_pb,
        mesh_scale=[1.0, 1.0, 1.0],
        base_pos=[0.45, 0, pos[-1]],
        base_ori=ori,
        rgba=[0.1, 1.0, 0.1, 0.25],
        mass=0.03)
    for jnt_id in range(p.getNumJoints(yumi_ar.arm.robot_id)):
        p.setCollisionFilterPair(yumi_ar.arm.robot_id, goal_obj_id, jnt_id, -1, enableCollision=False)    
    p.setCollisionFilterPair(
        yumi_ar.arm.robot_id, goal_obj_id, table_id, -1, enableCollision=True)

    p.setCollisionFilterPair(
        yumi_ar.arm.robot_id, goal_obj_id, r_gel_id, -1, enableCollision=False)
    p.setCollisionFilterPair(
        obj_id, goal_obj_id, -1, -1, enableCollision=False)
    p.setCollisionFilterPair(
        yumi_ar.arm.robot_id, obj_id, r_gel_id, -1, enableCollision=True)
    p.setCollisionFilterPair(
        yumi_ar.arm.robot_id, obj_id, 27, -1, enableCollision=True)

    primitive_name = args.primitive
    # face = np.random.randint(0)
    face = 0
    # face = 4

    mesh_file = stl_file
    cuboid_fname = mesh_file
    exp_single = SingleArmPrimitives(
        pull_cfg,
        yumi_ar.pb_client.get_client_id(),
        obj_id,
        mesh_file)
    # exp_single = SingleArmPrimitives(
    #     push_cfg,
    #     yumi_ar.pb_client.get_client_id(),
    #     obj_id,
    #     mesh_file)    

    exp_double = DualArmPrimitives(
        grasp_cfg,
        yumi_ar.pb_client.get_client_id(),
        obj_id,
        mesh_file,
        goal_face=face)
    exp_double.reset_graph(face)

    action_planner = ClosedLoopMacroActions(
        cfg,
        yumi_gs,
        obj_id,
        yumi_ar.pb_client.get_client_id(),
        args.config_package_path,
        replan=args.replan,
        object_mesh_file=mesh_file
    )

    trans_box_lock = threading.RLock()
    goal_viz = GoalVisual(
        trans_box_lock,
        goal_obj_id,
        action_planner.pb_client,
        cfg.OBJECT_POSE_3)

    multistep_planner = MultiStepManager(cfg, grasp_cfg, pull_cfg)

    action_planner.update_object(obj_id, mesh_file)
    exp_single.initialize_object(obj_id, cuboid_fname)
    print('Reset multi block!')
    # embed()

    for _ in range(args.num_obj_samples):
        full_args = multistep_planner.get_args(
            exp_single,
            exp_double,
            face,
            execute=False,
            skeleton=['pull', 'grasp', 'pull']
        )
            
        valid_subplans = 0
        valid_plans = []
        for plan_args in full_args:
            plan = action_planner.get_primitive_plan(
                plan_args['name'], plan_args, 'right')
            goal_viz.update_goal_state(
                util.pose_stamped2list(plan_args['object_pose2_world'])
            )

            start_pose = plan_args['object_pose1_world']
            goal_pose = plan_args['object_pose2_world']

            if args.debug:
                for _ in range(10):
                #     simulation.visualize_object(
                #         start_pose,
                #         filepath="package://config/descriptions/meshes/objects/cuboids/" +
                #         cuboid_fname.split('objects/cuboids')[1],
                #         name="/object_initial",
                #         color=(1., 0., 0., 1.),
                #         frame_id="/yumi_body",
                #         scale=(1., 1., 1.))
                #     simulation.visualize_object(
                #         goal_pose,
                #         filepath="package://config/descriptions/meshes/objects/cuboids/" +
                #         cuboid_fname.split('objects/cuboids')[1],
                #         name="/object_final",
                #         color=(0., 0., 1., 1.),
                #         frame_id="/yumi_body",
                #         scale=(1., 1., 1.))
                #     rospy.sleep(.1)
                # simulation.simulate(plan, cuboid_fname.split('objects/cuboids')[1])

                    simulation.visualize_object(
                        start_pose,
                        filepath="package://config/descriptions/meshes/objects/" +
                        cuboid_fname.split('objects/')[1],
                        name="/object_initial",
                        color=(1., 0., 0., 1.),
                        frame_id="/yumi_body",
                        scale=(1., 1., 1.))
                    simulation.visualize_object(
                        goal_pose,
                        filepath="package://config/descriptions/meshes/objects/" +
                        cuboid_fname.split('objects/')[1],
                        name="/object_final",
                        color=(0., 0., 1., 1.),
                        frame_id="/yumi_body",
                        scale=(1., 1., 1.))
                    rospy.sleep(.1)
                simulation.simulate(plan, cuboid_fname.split('objects/')[1])                

                #     simulation.visualize_object(
                #         start_pose,
                #         filepath="package://config/descriptions/meshes/objects/cylinders" +
                #         cuboid_fname.split('objects/cylinders')[1],
                #         name="/object_initial",
                #         color=(1., 0., 0., 1.),
                #         frame_id="/yumi_body",
                #         scale=(1., 1., 1.))
                #     simulation.visualize_object(
                #         goal_pose,
                #         filepath="package://config/descriptions/meshes/objects/cylinders" +
                #         cuboid_fname.split('objects/cylinders')[1],
                #         name="/object_final",
                #         color=(0., 0., 1., 1.),
                #         frame_id="/yumi_body",
                #         scale=(1., 1., 1.))
                #     rospy.sleep(.1)
                # simulation.simulate(plan, cuboid_fname.split('objects/cylinders')[1])                  
            else:
                primitive_name = plan_args['name']
                start_joints = {}
                if primitive_name == 'pull':
                    start_joints['right'] = pull_cfg.RIGHT_INIT
                    start_joints['left'] = pull_cfg.LEFT_INIT
                elif primitive_name == 'push':
                    start_joints['right'] = push_cfg.RIGHT_INIT
                    start_joints['left'] = push_cfg.LEFT_INIT                    
                elif primitive_name == 'grasp':
                    start_joints['right'] = grasp_cfg.RIGHT_INIT
                    start_joints['left'] = grasp_cfg.LEFT_INIT 
                yumi_ar.arm.set_jpos(start_joints['right']+start_joints['left'], ignore_physics=True)
                subplan_valid = action_planner.full_mp_check(
                    plan, 
                    primitive_name)
                if subplan_valid:
                    print("subplan valid!")
                    valid_subplans += 1
                    valid_plans.append(plan)
                else:
                    print("not valid, primitive: " + primitive_name)
                         
        if valid_subplans == len(full_args) and not args.debug:      
            for playback_num in range(2):
                # reset to initial state among all plans
                yumi_ar.pb_client.reset_body(
                    obj_id, 
                    util.pose_stamped2list(full_args[0]['object_pose1_world'])[:3],
                    util.pose_stamped2list(full_args[0]['object_pose1_world'])[3:])                
                if playback_num == 0:
                    # update goal state to final state among all plans
                    goal_viz.update_goal_state(
                        util.pose_stamped2list(full_args[-1]['object_pose2_world'])
                    ) 
                    goal_viz.show_goal_obj()
                else:
                    goal_viz.hide_goal_obj()                
                for plan_args in full_args:
                    if playback_num == 1:
                        goal_viz.hide_goal_obj()
                        goal_viz.update_goal_state(
                            util.pose_stamped2list(plan_args['object_pose2_world'])
                        ) 
                        goal_viz.show_goal_obj()                    
                    # teleport to start state, to avoid colliding with object during transit
                    primitive_name = plan_args['name']
                    print('executing primitive ' + str(primitive_name))
                    time.sleep(0.1)
                    try:
                        if 'left' in primitive_name:
                            arm = 'left'
                            action_planner.active_arm = 'left'
                            action_planner.inactive_arm = 'right'
                        else:
                            arm = 'right'
                            action_planner.active_arm = 'right'
                            action_planner.inactive_arm = 'left'
                        if 'push' in primitive_name:
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
                                rollingFriction=10e-4
                            )

                            p.changeDynamics(
                                yumi_ar.arm.robot_id,
                                l_gel_id,
                                rollingFriction=10e-4
                            )                            
                        if primitive_name in ['pull', 'push']:
                            # set arm configuration to good start state
                            if primitive_name == 'pull':
                                yumi_ar.arm.set_jpos(pull_cfg.RIGHT_INIT + pull_cfg.LEFT_INIT, ignore_physics=True)
                            else:
                                yumi_ar.arm.go_home(ignore_physics=True)
                            local_plan = action_planner.get_primitive_plan(primitive_name, plan_args, arm)
                            pull_plan = local_plan[0]

                            if 'push' in primitive_name:
                                action_planner.playback_single_arm('push', pull_plan, pre=True)
                            else:
                                # action_planner.playback_single_arm('push', pull_plan, pre=True)                                
                                _ = action_planner.execute('pull', plan_args)
                        elif 'grasp' in primitive_name:
                            yumi_ar.arm.set_jpos(grasp_cfg.RIGHT_INIT + grasp_cfg.LEFT_INIT, ignore_physics=True)
                            local_plan = action_planner.get_primitive_plan(primitive_name, plan_args, 'right')
                            grasp_plan = local_plan
                            for k, subplan in enumerate(grasp_plan):
                                action_planner.playback_dual_arm('grasp', subplan, k, pre=True)
                                time.sleep(1.0)
                    except ValueError as e:
                        print("Value error: ")
                        print(e)

                time.sleep(1.0)

            time.sleep(1.0)
            yumi_ar.arm.go_home(ignore_physics=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_trials',
        type=int
    )

    parser.add_argument(
        '--step_repeat',
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
        '--num_obj_samples', type=int, default=100
    )

    parser.add_argument(
        '--primitive',
        type=str,
        default='pull',
        help='which primitive to plan')

    args = parser.parse_args()
    main(args)
