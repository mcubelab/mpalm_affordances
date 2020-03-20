import os
import sys
import time
import argparse
import numpy as np
import rospy
import signal
import threading
from IPython import embed

from airobot import Robot
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions, YumiGelslimPybulet
from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

from closed_loop_experiments_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler
from data_gen_utils import YumiCamsGS, DataManager, MultiBlockManager, GoalVisual


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


def main(args):
    cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    rospy.init_node('MacroActions')
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

    yumi_gs = YumiGelslimPybulet(
        yumi_ar,
        cfg,
        exec_thread=False
    )

    for _ in range(10):
        yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    cuboid_sampler = CuboidSampler(
        '/root/catkin_ws/src/primitives/objects/cuboids/nominal_cuboid.stl',
        pb_client=yumi_ar.pb_client)
    cuboid_fname_template = '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids/'

    cuboid_manager = MultiBlockManager(
        cuboid_fname_template,
        cuboid_sampler,
        robot_id=yumi_ar.arm.robot_id,
        table_id=27,
        r_gel_id=r_gel_id,
        l_gel_id=l_gel_id)

    cuboid_fname = cuboid_manager.get_cuboid_fname()
    print("Cuboid file: " + cuboid_fname)
    # cuboid_fname = os.path.join(cuboid_fname_template, 'test_cuboid_smaller_4206.stl')

    obj_id, sphere_ids, mesh, goal_obj_id = \
        cuboid_sampler.sample_cuboid_pybullet(
            cuboid_fname,
            goal=True,
            keypoints=False)

    cuboid_manager.filter_collisions(obj_id, goal_obj_id)

    p.changeDynamics(
        obj_id,
        -1,
        lateralFriction=0.4
    )

    manipulated_object = None
    object_pose1_world = util.list2pose_stamped(cfg.OBJECT_INIT)
    object_pose2_world = util.list2pose_stamped(cfg.OBJECT_FINAL)
    palm_pose_l_object = util.list2pose_stamped(cfg.PALM_LEFT)
    palm_pose_r_object = util.list2pose_stamped(cfg.PALM_RIGHT)

    example_args = {}
    example_args['object_pose1_world'] = object_pose1_world
    example_args['object_pose2_world'] = object_pose2_world
    example_args['palm_pose_l_object'] = palm_pose_l_object
    example_args['palm_pose_r_object'] = palm_pose_r_object
    example_args['object'] = manipulated_object
    example_args['N'] = 60
    example_args['init'] = True
    example_args['table_face'] = 0

    primitive_name = args.primitive
    goal_face = 0

    # mesh_file = args.config_package_path + 'descriptions/meshes/objects/' + \
    #     args.object_name + '_experiments.stl'
    mesh_file = cuboid_fname
    exp_single = SingleArmPrimitives(
        cfg,
        yumi_ar.pb_client.get_client_id(),
        obj_id,
        mesh_file)
    k = 0
    while True:
        k += 1
        if k > 10:
            print('FAILED TO BUILD GRASPING GRAPH')
            return
        try:
            exp_double = DualArmPrimitives(
                cfg,
                yumi_ar.pb_client.get_client_id(),
                obj_id,
                mesh_file,
                goal_face=goal_face)
            break
        except ValueError as e:
            print(e)
            yumi_ar.pb_client.remove_body(obj_id)
            yumi_ar.pb_client.remove_body(goal_obj_id)
            cuboid_fname = cuboid_manager.get_cuboid_fname()
            print("Cuboid file: " + cuboid_fname)

            obj_id, sphere_ids, mesh, goal_obj_id = \
                cuboid_sampler.sample_cuboid_pybullet(
                    cuboid_fname,
                    goal=True,
                    keypoints=False)

            cuboid_manager.filter_collisions(obj_id, goal_obj_id)

            p.changeDynamics(
                obj_id,
                -1,
                lateralFriction=0.4)            
    if primitive_name == 'grasp':
        exp_running = exp_double
    else:
        exp_running = exp_single

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

    action_planner.update_object(obj_id, mesh_file)
    exp_single.initialize_object(obj_id, cuboid_fname)
    exp_double.initialize_object(obj_id, cuboid_fname, goal_face)
    print('Reset multi block!')

    for _ in range(args.num_blocks):
        for _ in range(args.num_obj_samples):
            if primitive_name == 'grasp':
                start_face = exp_double.get_valid_ind()
                if start_face is None:
                    print('Could not find valid start face')
                    continue
                plan_args = exp_double.get_random_primitive_args(ind=start_face,
                                                                 random_goal=True,
                                                                 execute=True)
            elif primitive_name == 'pull':
                plan_args = exp_single.get_random_primitive_args(ind=None,
                                                                 random_goal=True,
                                                                 execute=True)

            start_pose = plan_args['object_pose1_world']
            goal_pose = plan_args['object_pose2_world']
            goal_viz.update_goal_state(util.pose_stamped2list(goal_pose))
            if args.debug:
                import simulation

                plan = action_planner.get_primitive_plan(primitive_name, plan_args, 'right')

                for i in range(10):
                    simulation.visualize_object(
                        start_pose,
                        filepath="package://config/descriptions/meshes/objects/cuboids/" +
                            cuboid_fname.split('objects/cuboids')[1],
                        name="/object_initial",
                        color=(1., 0., 0., 1.),
                        frame_id="/yumi_body",
                        scale=(1., 1., 1.))
                    simulation.visualize_object(
                        goal_pose,
                        filepath="package://config/descriptions/meshes/objects/cuboids/" +
                            cuboid_fname.split('objects/cuboids')[1],
                        name="/object_final",
                        color=(0., 0., 1., 1.),
                        frame_id="/yumi_body",
                        scale=(1., 1., 1.))
                    rospy.sleep(.1)
                simulation.simulate(plan, cuboid_fname.split('objects/cuboids')[1])
            else:
                try:
                    result = action_planner.execute(primitive_name, plan_args)
                    if result is not None:
                        print(str(result[0]))

                except ValueError as e:
                    print("Value error: ")
                    print(e)

                if args.nice_pull_release:
                    time.sleep(1.0)

                    pose = util.pose_stamped2list(yumi_gs.compute_fk(yumi_gs.get_jpos(arm='right')))
                    pos, ori = pose[:3], pose[3:]

                    pos[2] += 0.001
                    r_jnts = yumi_gs.compute_ik(pos, ori, yumi_gs.get_jpos(arm='right'))
                    l_jnts = yumi_gs.get_jpos(arm='left')

                    if r_jnts is not None:
                        for _ in range(10):
                            pos[2] += 0.001
                            r_jnts = yumi_gs.compute_ik(pos, ori, yumi_gs.get_jpos(arm='right'))
                            l_jnts = yumi_gs.get_jpos(arm='left')

                            if r_jnts is not None:
                                yumi_gs.update_joints(list(r_jnts) + l_jnts)
                            time.sleep(0.1)

                time.sleep(0.1)
                for _ in range(10):
                    yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

        while True:
            try:
                yumi_ar.pb_client.remove_body(obj_id)
                yumi_ar.pb_client.remove_body(goal_obj_id)
                cuboid_fname = cuboid_manager.get_cuboid_fname()
                print("Cuboid file: " + cuboid_fname)

                obj_id, sphere_ids, mesh, goal_obj_id = \
                    cuboid_sampler.sample_cuboid_pybullet(
                        cuboid_fname,
                        goal=True,
                        keypoints=False)

                cuboid_manager.filter_collisions(obj_id, goal_obj_id)

                p.changeDynamics(
                    obj_id,
                    -1,
                    lateralFriction=0.4
                )
                action_planner.update_object(obj_id, mesh_file)
                exp_single.initialize_object(obj_id, cuboid_fname)
                exp_double.initialize_object(obj_id, cuboid_fname, goal_face)
                goal_viz.update_goal_obj(goal_obj_id)
                break
            except ValueError as e:
                print(e)



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

    args = parser.parse_args()
    main(args)
