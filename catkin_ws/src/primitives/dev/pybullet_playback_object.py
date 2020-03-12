from planning import pushing_planning, grasp_planning, levering_planning, pulling_planning
from helper import util, planning_helper, collisions

import os
from example_config import get_cfg_defaults

import airobot as ar
import pybullet as p
import time
import argparse
import numpy as np

import pickle

data = {}
data['palm_pose_world'] = []
data['object_pose_palm'] = []
data['contact_bool'] = []

def get_joint_poses(tip_poses, robot, cfg, nullspace=True):
    tip_to_wrist = util.list2pose_stamped(cfg.TIP_TO_WRIST_TF, '')
    world_to_world = util.unit_pose()

    r_joints, l_joints = None, None

    wrist_left = util.convert_reference_frame(
        tip_to_wrist,
        world_to_world,
        tip_poses[0],
        "yumi_body")
    wrist_right = util.convert_reference_frame(
        tip_to_wrist,
        world_to_world,
        tip_poses[1],
        "yumi_body")

    wrist_left = util.pose_stamped2list(wrist_left)
    wrist_right = util.pose_stamped2list(wrist_right)

    # r_joints = robot._accurate_ik(
    # 	wrist_right[0:3],
    # 	wrist_right[3:],
    # 	arm='right',
    # 	nullspace=nullspace)[:7]

    # l_joints = robot._accurate_ik(
    # 	wrist_left[0:3],
    # 	wrist_left[3:],
    # 	arm='left',
    # 	nullspace=nullspace)[7:]
    r_joints = robot.compute_ik(
        wrist_right[0:3],
        wrist_right[3:],
        arm='right',
        nullspace=nullspace)[:7]

    l_joints = robot.compute_ik(
        wrist_left[0:3],
        wrist_left[3:],
        arm='left',
        nullspace=nullspace)[7+2:-2]
    return r_joints, l_joints, wrist_right, wrist_left

def main(args):
    print(args)

    yumi = ar.create_robot('yumi',
                           robot_cfg={'render': True, 'self_collision': False})
    yumi.go_home()
    # while not yumi._reach_jnt_goal(yumi.cfgs.HOME_POSITION):
    # 	yumi.p.stepSimulation()
    # 	time.sleep(0.001)

    cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()
    print(cfg)

    manipulated_object = None
    object_pose1_world = util.list2pose_stamped(cfg.OBJECT_INIT)
    object_pose2_world = util.list2pose_stamped(cfg.OBJECT_FINAL)
    palm_pose_l_object = util.list2pose_stamped(cfg.PALM_LEFT)
    palm_pose_r_object = util.list2pose_stamped(cfg.PALM_RIGHT)

    if args.primitive == 'push':
        plan = pushing_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object)

    elif args.primitive == 'grasp':
        plan = grasp_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object)

    elif args.primitive == 'pivot':
        gripper_name = args.config_package_path + \
            'descriptions/meshes/mpalm/mpalms_all_coarse.stl'
        table_name = args.config_package_path + \
            'descriptions/meshes/table/table_top.stl'

        manipulated_object = collisions.CollisionBody(
            args.config_package_path + 'descriptions/meshes/objects/realsense_box_experiments.stl')

        plan = levering_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            gripper_name=gripper_name,
            table_name=table_name)

    elif args.primitive == 'pull':
        plan = pulling_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            arm='r')

    else:
        raise NotImplementedError

    object_loaded = False

    # plan_dict = plan[0]
    # i = 1
    # tip_poses = plan_dict['palm_poses_world'][i]

    # r_joints, l_joints = get_joint_poses(tip_poses, yumi, cfg, nullspace=True)

    # from IPython import embed
    # embed()
    with open('./data/' + args.primitive + '_object_poses_tip.pkl', 'rb') as f:
        data = pickle.load(f)

    # print("data: ")
    # print(data)

    object_poses_palm = data['object_pose_palm']
    palm_poses_world = data['palm_pose_world']
    object_poses_world = []
    for i, pose in enumerate(object_poses_palm):
        tmp_obj_pose = util.list2pose_stamped(pose)
        palm_pose = util.list2pose_stamped(palm_poses_world[i])

        tmp_obj_pose_world = util.convert_reference_frame(tmp_obj_pose, palm_pose, util.unit_pose())
        obj_pose_world = util.pose_stamped2list(tmp_obj_pose_world)

        object_poses_world.append(obj_pose_world)

    box_id = yumi.load_object(
        args.config_package_path+'descriptions/urdf/realsense_box.urdf',
        cfg.OBJECT_INIT[0:3],
        cfg.OBJECT_INIT[3:]
    )
    for i, pose in enumerate(object_poses_world):
        yumi.p.resetBasePositionAndOrientation(box_id, pose[:3], pose[3:])
        time.sleep(0.01)


    # new_pos_vec = []
    # des_pos_vec = []
    # for plan_dict in plan:
    #     for i, t in enumerate(plan_dict['t']):
    #         tip_poses = plan_dict['palm_poses_world'][i]

    #         r_joints, l_joints, wrist_right, wrist_left = get_joint_poses(tip_poses, yumi, cfg, nullspace=False)

    #         loop_time = 0.0875
    #         sleep_time = 0.005
    #         start = time.time()
            # if args.primitive != 'push':
            #     while (time.time() - start < loop_time):
            #         success = yumi.set_jpos(r_joints, arm='right', wait=False)
            #         time.sleep(sleep_time)
            #         yumi.set_jpos(l_joints, arm='left', wait=False)
            #         time.sleep(sleep_time)


                    # pts = yumi.p.getContactPoints(
                    #     bodyA=yumi.robot_id, bodyB=box_id, linkIndexA=12)
                    # contact_bool = 0 if len(pts) == 0 else 1

                    # object_pose_world = list(yumi.p.getBasePositionAndOrientation(box_id, 0)[
                    #     0]) + list(yumi.p.getBasePositionAndOrientation(box_id, 0)[1])
                    # object_pose_world = util.list2pose_stamped(object_pose_world)

                    # palm_frame_world = list(yumi.p.getLinkState(yumi.robot_id, 12)[
                    #                         0]) + list(yumi.p.getLinkState(yumi.robot_id, 12)[1])
                    # palm_frame_world = util.list2pose_stamped(palm_frame_world)

                    # object_pose_palm = util.convert_reference_frame(
                    #     object_pose_world, util.unit_pose(), palm_frame_world)

                    # data['palm_pose_world'].append(
                    #     util.pose_stamped2list(tip_poses[1]))
                    # data['object_pose_palm'].append(
                    #     util.pose_stamped2list(object_pose_palm))
                    # data['contact_bool'].append(contact_bool)
            # else:
            #     while (time.time() - start < loop_time):
            #         yumi.set_jpos(r_joints, arm='right', wait=False)
            #         time.sleep(sleep_time)

            # r_success = yumi.set_jpos(r_joints, arm='right', wait=True)
            # l_success = yumi.set_jpos(l_joints, arm='left', wait=True)

            # r_success = yumi.set_ee_pose(wrist_right[0:3], wrist_right[3:], arm='right')
            # l_success = yumi.set_ee_pose(wrist_left[0:3], wrist_left[3:], arm='left')

            # new_pos = yumi.get_ee_pose(arm='left')[0]
            # new_pos_vec.append(new_pos)
            # des_pos_vec.append(wrist_left[:3])

            # if not (r_success and l_success):
            # if not r_success:
            # if not l_success:
            # 	print("i: ", i)
            # 	from IPython import embed
            # 	embed()
            # 	break
            # time.sleep(0.05)
            # yumi.p.stepSimulation()

            # if i == 0 and not object_loaded and args.object:
            #     time.sleep(2.0)
            #     box_id = yumi.load_object(
            #         args.config_package_path+'descriptions/urdf/realsense_box.urdf',
            #         cfg.OBJECT_INIT[0:3],
            #         cfg.OBJECT_INIT[3:]
            #     )

            #     yumi.p.setGravity(0, 0, -9.8)
            #     time.sleep(2.0)
            #     object_loaded = True

                # from IPython import embed
                # embed()

            # pts = yumi.p.getContactPoints(
            #     bodyA=yumi.robot_id, bodyB=box_id, linkIndexA=12)
            # contact_bool = 0 if len(pts) == 0 else 1

            # from IPython import embed
            # embed()

            # object_pose_world = list(yumi.p.getBasePositionAndOrientation(box_id, 0)[
            #                          0]) + list(yumi.p.getBasePositionAndOrientation(box_id, 0)[1])
            # object_pose_world = util.list2pose_stamped(object_pose_world)

            # palm_frame_world = list(yumi.p.getLinkState(yumi.robot_id, 12)[
            #                         0]) + list(yumi.p.getLinkState(yumi.robot_id, 12)[1])
            # palm_frame_world = util.list2pose_stamped(palm_frame_world)

            # object_pose_palm = util.convert_reference_frame(
            #     object_pose_world, util.unit_pose(), palm_frame_world)

            # data['palm_pose_world'].append(util.pose_stamped2list(tip_poses[1]))
            # data['object_pose_palm'].append(util.pose_stamped2list(object_pose_palm))
            # data['contact_bool'].append(contact_bool)

    # with open(args.primitive+'_object_poses.pkl', 'wb') as f:
    #     pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_package_path',
                        type=str,
                         default='/home/anthony/repos/research/mpalm_affordances/catkin_ws/src/config/')
    parser.add_argument('--example_config_path', type=str, default='config')
    parser.add_argument('--primitive', type=str, default='push', help='which primitive to plan')
    parser.add_argument('--simulate', type=int, default=1)
    parser.add_argument('--object', type=int, default=0)
    args = parser.parse_args()
    main(args)
