from planning import pushing_planning, grasp_planning, levering_planning, pulling_planning
from helper import util, planning_helper, collisions

import os
# from example_config import get_cfg_defaults
from closed_loop_experiments import get_cfg_defaults

from airobot import Robot
from airobot.utils import pb_util, common, arm_util
import pybullet as p
import time
import argparse
import numpy as np

import pickle
import rospy
from IPython import embed

import trimesh
import copy

from macro_actions import ClosedLoopMacroActions, YumiGelslimPybulet


class EvalPrimitives():
    def __init__(self, cfg, object_id, mesh_file):
        self.cfg = cfg
        self.object_id = object_id

        self.init_poses = [
            self.cfg.OBJECT_POSE_1,
            self.cfg.OBJECT_POSE_2,
            self.cfg.OBJECT_POSE_3
        ]

        self.init_oris = []
        for i, pose in enumerate(self.init_poses):
            self.init_oris.append(pose[3:])

        self.pb_client = pb_util.PB_CLIENT

        self.x_bounds = [0.2, 0.55]
        self.y_bounds = [-0.4, -0.01]
        self.default_z = 0.1

        self.mesh_file = mesh_file
        self.mesh = trimesh.load(self.mesh_file)
        self.mesh_world = copy.deepcopy(self.mesh)

    def transform_mesh_world(self):
        self.mesh_world = copy.deepcopy(self.mesh)
        obj_pos_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[0])
        obj_ori_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[1])
        obj_ori_mat = common.quat2rot(obj_ori_world)
        h_trans = np.zeros((4, 4))
        h_trans[:3, :3] = obj_ori_mat
        h_trans[:-1, -1] = obj_pos_world
        h_trans[-1, -1] = 1
        self.mesh_world.apply_transform(h_trans)

    def get_obj_pose(self):
        obj_pos_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[0])
        obj_ori_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[1])

        obj_pose_world = util.list2pose_stamped(obj_pos_world + obj_ori_world)
        return obj_pose_world, obj_pos_world + obj_ori_world

    def get_rand_init(self, execute=True):
        rand_yaw = np.pi*np.random.random_sample()
        dq = common.euler2quat([0, 0, rand_yaw]).tolist()
        init_ind = np.random.randint(len(self.init_oris))
        q = common.quat_multiply(
            dq,
            self.init_oris[init_ind])
        x = self.x_bounds[0] + (self.x_bounds[1] - self.x_bounds[0]) * np.random.random_sample()
        y = self.y_bounds[0] + (self.y_bounds[1] - self.y_bounds[0]) * np.random.random_sample()
        if execute:
            p.resetBasePositionAndOrientation(
                self.object_id,
                [x, y, self.default_z],
                q,
                self.pb_client)
        
        time.sleep(1.0)
        self.transform_mesh_world()
        return x, y, q, init_ind

    def get_init(self, ind, execute=True):
        if execute:
            p.resetBasePositionAndOrientation(
                self.object_id,
                self.init_poses[ind][:3],
                self.init_poses[ind][3:],
                self.pb_client
            )
        return self.init_poses[ind]

    def sample_contact(self, primitive_name='push', N=1):
        valid = False
        timeout = 10
        start = time.time()
        while not valid:
            sampled_contact, sampled_facet = self.mesh_world.sample(N, True)
            sampled_normal = self.mesh_world.face_normals[sampled_facet[0]]
            if primitive_name == 'push':
                in_xy = np.abs(np.dot(sampled_normal, [0, 0, 1])) < 0.0001

                if in_xy:
                    valid = True
            
            elif primitive_name == 'pull':
                parallel_z = np.abs(np.dot(sampled_normal, [1, 0, 0])) < 0.0001 and \
                    np.abs(np.dot(sampled_normal, [0, 1, 0])) < 0.0001

                above_com = sampled_contact[0][-1] > self.mesh_world.center_mass[-1]
                
                if parallel_z and above_com:
                    valid = True
            else:
                raise ValueError('Primitive name not recognized')
            
            if time.time() - start > timeout:
                print("Contact point sample timed out! Exiting")
                return None, None, None

        return sampled_contact, sampled_normal, sampled_facet


# class YumiExecution():
#     def __init__(self, yumi_ar):
#         self.yumi_ar = yumi_ar
    
#     def exec_pull(self)


def main(args):
    cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    rospy.init_node('MacroActions')

    # setup yumi
    yumi_ar = Robot('yumi',
                    pb=True,
                    arm_cfg={'render': True, 'self_collision': False})
    yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    # setup yumi_gs
    yumi_gs = YumiGelslimPybulet(yumi_ar, cfg, exec_thread=True)

    # embed()

    if args.object:
        # box_id = pb_util.load_urdf(
        #     args.config_package_path +
        #     'descriptions/urdf/'+args.object_name+'.urdf',
        #     cfg.OBJECT_INIT[0:3],
        #     cfg.OBJECT_INIT[3:]
        # )
        box_id = pb_util.load_urdf(
            args.config_package_path +
            'descriptions/urdf/'+args.object_name+'_trans.urdf',
            cfg.OBJECT_POSE_3[0:3],
            cfg.OBJECT_POSE_3[3:]
        )        

    # setup macro_planner
    action_planner = ClosedLoopMacroActions(
        cfg,
        yumi_gs,
        box_id,
        pb_util.PB_CLIENT,
        args.config_package_path,
        replan=args.replan
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
    example_args['N'] = 60  # 60
    example_args['init'] = True

    primitive_name = args.primitive

    mesh_file = args.config_package_path + 'descriptions/meshes/objects/' + args.object_name + '_experiments.stl'
    exp = EvalPrimitives(cfg, box_id, mesh_file)

    # embed()

    while True:
        init_id = exp.get_rand_init()[-1]
        if init_id == 0:
            break
        time.sleep(0.01)
    point, normal, face = exp.sample_contact('pull')
    # if point is not None:
    #     delta_xyz = point[0] - yumi_ar.arm.get_ee_pose(arm='right')[0]
    #     approach_xyz = delta_xyz
    #     approach_xyz[-1] += 0.1
    #     yumi_ar.arm.move_ee_xyz(approach_xyz, arm='right')
    #     yumi_ar.arm.move_ee_xyz([0, 0, -0.1], arm='right')
    #     time.sleep(1.0)
    #     yumi_ar.arm.move_ee_xyz([0, -0.2, 0], arm='right')
    #     time.sleep(1.0)
    #     yumi_ar.arm.move_ee_xyz([0, 0, 0.1], arm='right')
    #     yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT)
    
    # wrist_pos = yumi_ar.arm.get_ee_pose(arm='right')[0]

    rand_yaw = (np.pi/2)*np.random.random_sample() + np.pi/4
    rand_ori = common.euler2quat([np.pi/2, 0, rand_yaw])

    # yumi_ar.arm.set_ee_pose(wrist_pos, rand_ori, arm='right')

    point_list = point[0].tolist()
    ori_list = rand_ori.tolist()

    world_pose_list = point_list + ori_list
    world_pose = util.list2pose_stamped(world_pose_list)

    obj_pos_world = list(p.getBasePositionAndOrientation(box_id, pb_util.PB_CLIENT)[0])
    obj_ori_world = list(p.getBasePositionAndOrientation(box_id, pb_util.PB_CLIENT)[1])

    obj_pose_world = util.list2pose_stamped(obj_pos_world + obj_ori_world)
    contact_obj_frame = util.convert_reference_frame(world_pose, obj_pose_world, util.unit_pose())
    example_args['palm_pose_r_object'] = contact_obj_frame
    example_args['object_pose1_world'] = obj_pose_world

    result = action_planner.execute(primitive_name, example_args)

    embed()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()
    main(args)