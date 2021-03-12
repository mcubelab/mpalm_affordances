import os, sys
import os.path as osp
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

import trimesh
import open3d
import pcl
import pybullet as p

import copy
import time
from IPython import embed

from yacs.config import CfgNode as CN
from airobot.utils import common

from rpo_planning.utils import common as util
from rpo_planning.utils.exceptions import PlanWaypointsError
from rpo_planning.utils.perception import registration as reg
from rpo_planning.utils.planning.skill import PrimitiveSkill
from rpo_planning.utils.planning.pointcloud_plan import PointCloudNode

# from closed_loop_experiments_cfg import get_cfg_defaults
# from eval_utils.visualization_tools import correct_grasp_pos, project_point2plane
# from pointcloud_planning_utils import PointCloudNode, PointCloudNodeForward
# from skill_utils import PrimitiveSkill, StateValidity


class GraspSkill(PrimitiveSkill):
    def __init__(self, sampler, robot, get_plan_func, ignore_mp=False, pp=False, avoid_collisions=True):
        super(GraspSkill, self).__init__(sampler, robot)
        self.x_min, self.x_max = 0.35, 0.45
        # self.y_min, self.y_max = -0.175, 0.175
        self.y_min, self.y_max = -0.1, 0.1
        self.start_joints = [0.9936, -2.1848, -0.9915, 0.8458, 3.7618,  1.5486,  0.1127,
                            -1.0777, -2.1187, 0.995, 1.002 ,  -3.6834,  1.8132,  2.6405]
        self.get_plan_func = get_plan_func
        self.ignore_mp = ignore_mp
        self.pick_and_place = pp
        self.avoid_collisions = avoid_collisions
        self.skill_name = 'grasp' if not pp else 'grasp-pp'

    def get_nominal_plan(self, plan_args):
        # from planning import grasp_planning_wf
        palm_pose_l_world = plan_args['palm_pose_l_world']
        palm_pose_r_world = plan_args['palm_pose_r_world']
        transformation = plan_args['transformation']
        N = plan_args['N']

        nominal_plan = self.get_plan_func(
            palm_pose_l_world=palm_pose_l_world,
            palm_pose_r_world=palm_pose_r_world,
            transformation=transformation,
            N=N
        )

        return nominal_plan

    def within_pap_margin(self, transformation):
        # euler = common.rot2euler(transformation[:-1, :-1])
        euler = R.from_dcm(transformation[:-1, :-1]).as_euler('xyz')
        # print('euler: ', euler)
        return np.abs(euler[0]) < np.deg2rad(20) and np.abs(euler[1]) < np.deg2rad(20)

    def valid_transformation(self, state):
        # TODO: check if too much roll
        if self.pick_and_place:
            valid = self.within_pap_margin(state.transformation)
        else:
            valid = True
        return valid

    def sample(self, state, target_surface=None, final_trans=False):
        # NN sampling, point cloud alignment
        if final_trans:
            prediction = self.sampler.sample(
                state=state.pointcloud,
                state_full=state.pointcloud_full,
                target=target_surface,
                final_trans_to_go=state.transformation_to_go)
        else:
            prediction = self.sampler.sample(
                state=state.pointcloud,
                state_full=state.pointcloud_full,
                target=target_surface,
                pp=self.pick_and_place,
                planes=state.planes)
        transformation = prediction['transformation']
        new_state = PointCloudNode()
        new_state.init_state(state, transformation, skill=self.skill_name)
        new_state.init_palms(prediction['palms'],
                             correction=True,
                             prev_pointcloud=state.pointcloud_full)
        # new_state = PointCloudNodeForward()
        # new_state.init_palms(prediction['palms'],
        #                      correction=True,
        #                      prev_pointcloud=state.pointcloud_full)
        # new_state.init_state(state, transformation, prediction['palms'])
        return new_state

    def satisfies_preconditions(self, state):
        # test 1: on the table
        valid = self.object_is_on_table(state)

        # test 2: in front of the robot
        valid = valid and self.object_in_grasp_region(state)
        return valid

    def object_in_grasp_region(self, state):
        # checks if the CoM is in a nice region in front of the robot
        pos = np.mean(state.pointcloud, axis=0)[0:2]
        x, y = pos[0], pos[1]
        x_valid = x < self.x_max and x > self.x_min
        y_valid = y < self.y_max and y > self.y_min
        return x_valid and y_valid

    def feasible_motion(self, state, start_joints=None, nominal_plan=None, avoid_collisions=None, jump_thresh=None):
        if avoid_collisions is None:
            avoid_collisions = self.avoid_collisions
        if self.ignore_mp:
            return True
        if nominal_plan is None:
            # construct plan args
            plan_args = {}
            plan_args['palm_pose_l_world'] = util.list2pose_stamped(
                state.palms[7:].tolist())
            plan_args['palm_pose_r_world'] = util.list2pose_stamped(
                state.palms[:7].tolist()
            )
            plan_args['transformation'] = util.pose_from_matrix(state.transformation)
            plan_args['N'] = 60

            # get primitive plan
            nominal_plan = self.get_nominal_plan(plan_args)

        right_valid = []
        left_valid = []

        if start_joints is None:
            last_joints_right, last_joints_left = self.start_joints[:7], self.start_joints[7:]
        else:
            last_joints_right, last_joints_left = start_joints['left'], start_joints['right']
        for subplan_number, subplan_dict in enumerate(nominal_plan):
            subplan_tip_poses = subplan_dict['palm_poses_world']

            # setup motion planning request with all the cartesian waypoints
            tip_right = []
            tip_left = []

            # bump y a bit in the palm frame for pre pose, for collision avoidance
            if subplan_number == 0:
                # pre_pose_right_init = util.unit_pose()
                # pre_pose_left_init = util.unit_pose()

                # pre_pose_right_init.pose.position.y += 0.05
                # pre_pose_left_init.pose.position.y += 0.05

                # pre_pose_right = util.transform_pose(
                #     pre_pose_right_init, subplan_tip_poses[0][1])

                # pre_pose_left = util.transform_pose(
                #     pre_pose_left_init, subplan_tip_poses[0][0])

                # tip_right.append(pre_pose_right.pose)
                # tip_left.append(pre_pose_left.pose)
                initial_pose = {}
                initial_pose['right'] = subplan_tip_poses[0][1]
                initial_pose['left'] = subplan_tip_poses[0][0]
                palm_y_normals = self.robot.get_palm_y_normals(palm_poses=initial_pose)
                normal_dir_r = (util.pose_stamped2np(palm_y_normals['right'])[:3] - util.pose_stamped2np(initial_pose['right'])[:3]) * 0.05
                normal_dir_l = (util.pose_stamped2np(palm_y_normals['left'])[:3] - util.pose_stamped2np(initial_pose['left'])[:3]) * 0.05

                pre_pose_right_pos = util.pose_stamped2np(initial_pose['right'])[:3] + normal_dir_r
                pre_pose_left_pos = util.pose_stamped2np(initial_pose['left'])[:3] + normal_dir_l

                pre_pose_right_np = np.hstack([pre_pose_right_pos, util.pose_stamped2np(initial_pose['right'])[3:]])
                pre_pose_left_np = np.hstack([pre_pose_left_pos, util.pose_stamped2np(initial_pose['left'])[3:]])
                pre_pose_right = util.list2pose_stamped(pre_pose_right_np)
                pre_pose_left = util.list2pose_stamped(pre_pose_left_np)  

                tip_right.append(pre_pose_right.pose)
                tip_left.append(pre_pose_left.pose)              

            for i in range(len(subplan_tip_poses)):
                tip_right.append(subplan_tip_poses[i][1].pose)
                tip_left.append(subplan_tip_poses[i][0].pose)

            # if start_joints is None:
            #     # l_start = self.robot.get_jpos(arm='left')
            #     # r_start = self.robot.get_jpos(arm='right')
            #     l_start = self.start_joints[7:]
            #     r_start = self.start_joints[:7]
            # else:
            #     l_start = start_joints['left']
            #     r_start = start_joints['right']
            l_start = last_joints_left
            r_start = last_joints_right

            try:
                r_plan = self.robot.mp_right.plan_waypoints(
                    tip_right,
                    force_start=l_start+r_start,
                    avoid_collisions=avoid_collisions
                )
                right_valid.append(1)
                last_joints_right = r_plan.points[-1].positions
            except PlanWaypointsError:
                break
            try:
                l_plan = self.robot.mp_left.plan_waypoints(
                    tip_left,
                    force_start=l_start+r_start,
                    avoid_collisions=avoid_collisions
                )
                left_valid.append(1)
                last_joints_left = l_plan.points[-1].positions
            except PlanWaypointsError:
                break
        valid = False
        if sum(right_valid) == len(nominal_plan) and \
                sum(left_valid) == len(nominal_plan):
            valid = True
        return valid


class PullRightSkill(PrimitiveSkill):
    def __init__(self, sampler, robot, get_plan_func, ignore_mp=False, avoid_collisions=True, visualize=False):
        super(PullRightSkill, self).__init__(sampler, robot)
        self.get_plan_func = get_plan_func
        self.start_joints_r = [0.417, -1.038, -1.45, 0.26, 0.424, 1.586, 2.032]
        self.start_joints_l = [-0.409, -1.104, 1.401, 0.311, -0.403, 1.304, 1.142]
        self.unit_n = 100
        self.ignore_mp = ignore_mp
        self.avoid_collisions = avoid_collisions
        self.visualize = visualize
        self.skill_name = 'pull_right'

    def get_nominal_plan(self, plan_args):
        # from planning import grasp_planning_wf
        palm_pose_l_world = plan_args['palm_pose_l_world']
        palm_pose_r_world = plan_args['palm_pose_r_world']
        transformation = plan_args['transformation']
        N = plan_args['N']

        nominal_plan = self.get_plan_func(
            palm_pose_l_world=palm_pose_l_world,
            palm_pose_r_world=palm_pose_r_world,
            transformation=transformation,
            N=N
        )

        return nominal_plan

    def valid_transformation(self, state):
        return self.within_se2_margin(state.transformation)

    def sample(self, state, *args, **kwargs):
        final_trans = False
        if 'final_trans' in kwargs.keys():
            final_trans = kwargs['final_trans']
        if final_trans:
            final_trans_to_go = state.transformation_to_go
        else:
            final_trans_to_go = None

        pcd_pts = state.pointcloud
        pcd_pts_full = None
        if state.pointcloud_full is not None:
            pcd_pts_full = state.pointcloud_full

        prediction = self.sampler.sample(
            pcd_pts,
            state_full=pcd_pts_full,
            final_trans_to_go=final_trans_to_go)
        new_state = PointCloudNode()
        # new_state.init_palms(prediction['palms'],
        #                      correction=True,
        #                      prev_pointcloud=pcd_pts_full,
        #                      dual=False)
        new_state.init_state(state, prediction['transformation'], skill=self.skill_name)
        # new_state.init_state(state, prediction['transformation'], prediction['palms'])

        # new_state.init_palms(prediction['palms'])
        new_state.init_palms(prediction['palms'],
                             correction=True,
                             prev_pointcloud=pcd_pts_full,
                             dual=False)
        
        if self.visualize:
            from rpo_planning.config.multistep_eval_cfg import get_multistep_cfg_defaults
            from rpo_planning.utils.visualize import PalmVis
            cfg = get_multistep_cfg_defaults()
            palm_mesh_file='/root/catkin_ws/src/config/descriptions/meshes/mpalm/mpalms_all_coarse.stl'
            table_mesh_file = '/root/catkin_ws/src/config/descriptions/meshes/table/table_top.stl'
            viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)        

            viz_data = {}
            viz_data['contact_world_frame_right'] = new_state.palms_raw[:7]
            viz_data['contact_world_frame_left'] = new_state.palms_raw[:7:]
            viz_data['transformation'] = util.pose_stamped2np(util.pose_from_matrix(prediction['transformation']))
            viz_data['object_pointcloud'] = pcd_pts_full
            viz_data['start'] = pcd_pts_full

            scene_pcd = viz_palms.vis_palms_pcd(viz_data, world=True, corr=False, full_path=True, show_mask=False, goal_number=1)
            scene_pcd.show()

            viz_data = {}
            viz_data['contact_world_frame_right'] = new_state.palms_corrected[:7]
            viz_data['contact_world_frame_left'] = new_state.palms_corrected[:7:]
            viz_data['transformation'] = util.pose_stamped2np(util.pose_from_matrix(prediction['transformation']))
            viz_data['object_pointcloud'] = pcd_pts_full
            viz_data['start'] = pcd_pts_full

            scene_pcd = viz_palms.vis_palms_pcd(viz_data, world=True, corr=False, full_path=True, show_mask=False, goal_number=1)
            scene_pcd.show()

            embed()

        return new_state

    def satisfies_preconditions(self, state):
        # test 1: on the table
        valid = self.object_is_on_table(state)
        return valid

    def calc_n(self, dx, dy):
        dist = np.sqrt(dx**2 + dy**2)
        N = max(2, int(dist*self.unit_n))
        return N

    def within_se2_margin(self, transformation):
        # euler = common.rot2euler(transformation[:-1, :-1])
        euler = R.from_dcm(transformation[:-1, :-1]).as_euler('xyz')
        # print('euler: ', euler)
        return np.abs(euler[0]) < np.deg2rad(20) and np.abs(euler[1]) < np.deg2rad(20)

    def feasible_motion(self, state, start_joints=None, nominal_plan=None, avoid_collisions=None, jump_thresh=None):
        if avoid_collisions is None:
            avoid_collisions = self.avoid_collisions
        # # check if transformation is within margin of pure SE(2) transformation
        if not self.within_se2_margin(state.transformation):
            return False

        if self.ignore_mp:
            return True

        # construct plan args
        if nominal_plan is None:
            plan_args = {}
            # just copying the right to the left, cause it's not being used anyways
            plan_args['palm_pose_l_world'] = util.list2pose_stamped(
                state.palms[:7].tolist())
            plan_args['palm_pose_r_world'] = util.list2pose_stamped(
                state.palms[:7].tolist()
            )
            plan_args['transformation'] = util.pose_from_matrix(state.transformation)
            plan_args['N'] = self.calc_n(state.transformation[0, -1],
                                         state.transformation[1, -1])

            # get primitive plan
            nominal_plan = self.get_nominal_plan(plan_args)

        subplan_tip_poses = nominal_plan[0]['palm_poses_world']

        # setup motion planning request with cartesian waypoints
        tip_right, tip_left = [], []

        # create an approach waypoint near the object
        pre_pose_right_init = util.unit_pose()
        pre_pose_left_init = util.unit_pose()

        pre_pose_right_init.pose.position.y += 0.05
        pre_pose_left_init.pose.position.y += 0.05

        pre_pose_right = util.transform_pose(pre_pose_right_init,
                                             subplan_tip_poses[0][1])
        pre_pose_left = util.transform_pose(pre_pose_left_init,
                                            subplan_tip_poses[0][0])
        tip_right.append(pre_pose_right.pose)
        tip_left.append(pre_pose_left.pose)

        # create all other cartesian waypoints
        for i in range(len(subplan_tip_poses)):
            tip_right.append(subplan_tip_poses[i][1].pose)
            tip_left.append(subplan_tip_poses[i][0].pose)

        if start_joints is None:
            r_start = self.start_joints_r
            l_start = self.start_joints_l
        else:
            r_start = start_joints['right']
            l_start = start_joints['left']

        # l_start = self.robot.get_jpos(arm='left')

        # plan cartesian path
        valid = False
        try:
            # self.robot.mp_right.plan_waypoints(
            #     tip_right,
            #     force_start=l_start+r_start,
            #     avoid_collisions=True
            # )
            self.mp_func(
                tip_right,
                tip_left,
                force_start=l_start+r_start,
                avoid_collisions=avoid_collisions,
                jump_thresh=jump_thresh
            )
            valid = True
        except PlanWaypointsError:
            pass

        return valid

    def mp_func(self, tip_right, tip_left, force_start, avoid_collisions=None, jump_thresh=None):
        if self.avoid_collisions is None:
            avoid_collisions = self.avoid_collisions
        self.robot.mp_right.plan_waypoints(
            tip_right,
            force_start=force_start,
            avoid_collisions=avoid_collisions,
            jump_thresh=jump_thresh
        )


class PullLeftSkill(PullRightSkill):
    def __init__(self, sampler, robot, get_plan_func, ignore_mp=False, avoid_collisions=True):
        super(PullLeftSkill, self).__init__(sampler, robot, get_plan_func, ignore_mp, avoid_collisions)
        self.skill_name = 'pull_left'

    def sample(self, state, *args, **kwargs):
        final_trans = False
        if 'final_trans' in kwargs.keys():
            final_trans = kwargs['final_trans']
        if final_trans:
            final_trans_to_go = state.transformation_to_go
        else:
            final_trans_to_go = None

        pcd_pts = copy.deepcopy(state.pointcloud)
        pcd_pts[:, 1] = -pcd_pts[:, 1]
        pcd_pts_full = None
        if state.pointcloud_full is not None:
            pcd_pts_full = copy.deepcopy(state.pointcloud_full)
            pcd_pts_full[:, 1] = -pcd_pts_full[:, 1]

        prediction = self.sampler.sample(
            pcd_pts,
            state_full=pcd_pts_full,
            final_trans_to_go=final_trans_to_go)

        if final_trans:
            # on last step we have to trust that this is correct
            new_transformation = prediction['transformation']
        else:
            # if in the middle, then flip based on the right pull transform
            new_transformation = copy.deepcopy(prediction['transformation'])
            new_transformation[0, 1] *= -1
            new_transformation[1, 0] *= -1
            new_transformation[1, -1] *= -1
        new_palms = util.pose_stamped2np(util.flip_palm_pulling(util.list2pose_stamped(prediction['palms'][:7])))
        new_palms[1] *= -1

        new_state = PointCloudNode()
        new_state.init_state(state, new_transformation, skill=self.skill_name)
        # new_state.init_palms(new_palms)
        new_state.init_palms(new_palms,
                             correction=True,
                             prev_pointcloud=state.pointcloud_full,
                             dual=False)
        return new_state

    def mp_func(self, tip_right, tip_left, force_start, avoid_collisions=None, jump_thresh=None):
        if avoid_collisions is None:
            avoid_collisions = self.avoid_collisions
        self.robot.mp_left.plan_waypoints(
            tip_left,
            force_start=force_start,
            avoid_collisions=avoid_collisions,
            jump_thresh=jump_thresh
        )


class PushRightSkill(PrimitiveSkill):
    def __init__(self, sampler, robot, get_plan_func, ignore_mp=False, avoid_collisions=True):
        super(PushRightSkill, self).__init__(sampler, robot)
        self.get_plan_func = get_plan_func
        self.start_joints_r = [0.413, -1.325, -1.040, -0.053, -0.484, 0.841, -1.546]
        self.start_joints_l = [-0.473, -1.450, 1.091, 0.031, 0.513, 0.77, -1.669]
        self.unit_n = 100
        self.ignore_mp = ignore_mp
        self.avoid_collisions = avoid_collisions

        # set pushing velocity a little high for planning speed up (execution is interpolated more densely)
        self.velocity_real = 0.05
        self.skill_name = 'push_right'

    def get_nominal_plan(self, plan_args):
        # from planning import grasp_planning_wf
        palm_pose_l_world = plan_args['palm_pose_l_world']
        palm_pose_r_world = plan_args['palm_pose_r_world']
        transformation = plan_args['transformation']
        N = plan_args['N']

        nominal_plan = self.get_plan_func(
            palm_pose_l_world=palm_pose_l_world,
            palm_pose_r_world=palm_pose_r_world,
            transformation=transformation,
            velocity_real=self.velocity_real
        )

        return nominal_plan

    def valid_transformation(self, state):
        return self.within_se2_margin(state.transformation)

    def sample(self, state, *args, **kwargs):
        final_trans = False
        if 'final_trans' in kwargs.keys():
            final_trans = kwargs['final_trans']
        if final_trans:
            final_trans_to_go = state.transformation_to_go
        else:
            final_trans_to_go = None

        pcd_pts = state.pointcloud
        pcd_pts_full = None
        if state.pointcloud_full is not None:
            pcd_pts_full = state.pointcloud_full

        prediction = self.sampler.sample(
            pcd_pts,
            state_full=pcd_pts_full,
            final_trans_to_go=final_trans_to_go)
        new_state = PointCloudNode()
        new_state.init_state(state, prediction['transformation'], skill=self.skill_name)
        new_state.init_palms(prediction['palms'],
                             correction=True,
                             prev_pointcloud=pcd_pts_full,
                             dual=False)
        return new_state

    def satisfies_preconditions(self, state):
        # test 1: on the table
        valid = self.object_is_on_table(state)
        return valid

    def calc_n(self, dx, dy):
        dist = np.sqrt(dx**2 + dy**2)
        N = max(2, int(dist*self.unit_n))
        return N

    def within_se2_margin(self, transformation):
        # euler = common.rot2euler(transformation[:-1, :-1])
        euler = R.from_dcm(transformation[:-1, :-1]).as_euler('xyz')
        # print('euler: ', euler)
        return np.abs(euler[0]) < np.deg2rad(20) and np.abs(euler[1]) < np.deg2rad(20)

    def feasible_motion(self, state, start_joints=None, nominal_plan=None, avoid_collisions=None, jump_thresh=None):
        if avoid_collisions is None:
            avoid_collisions = self.avoid_collisions

        # # check if transformation is within margin of pure SE(2) transformation
        if not self.within_se2_margin(state.transformation):
            return False

        if self.ignore_mp:
            return True

        # construct plan args
        if nominal_plan is None:
            plan_args = {}
            # just copying the right to the left, cause it's not being used anyways
            plan_args['palm_pose_l_world'] = util.list2pose_stamped(
                state.palms[:7].tolist())
            plan_args['palm_pose_r_world'] = util.list2pose_stamped(
                state.palms[:7].tolist()
            )
            plan_args['transformation'] = util.pose_from_matrix(state.transformation)
            plan_args['N'] = self.calc_n(state.transformation[0, -1],
                                         state.transformation[1, -1])

            # get primitive plan
            nominal_plan = self.get_nominal_plan(plan_args)

        subplan_tip_poses = nominal_plan[0]['palm_poses_world']

        # setup motion planning request with cartesian waypoints
        tip_right, tip_left = [], []

        # create an approach waypoint near the object
        pre_pose_right_init = util.unit_pose()
        pre_pose_left_init = util.unit_pose()

        pre_pose_right_init.pose.position.y += 0.05
        pre_pose_left_init.pose.position.y += 0.05

        pre_pose_right = util.transform_pose(pre_pose_right_init,
                                             subplan_tip_poses[0][1])
        pre_pose_left = util.transform_pose(pre_pose_left_init,
                                            subplan_tip_poses[0][0])
        tip_right.append(pre_pose_right.pose)
        tip_left.append(pre_pose_left.pose)

        # create all other cartesian waypoints
        for i in range(len(subplan_tip_poses)):
            tip_right.append(subplan_tip_poses[i][1].pose)
            tip_left.append(subplan_tip_poses[i][0].pose)

        if start_joints is None:
            r_start = self.start_joints_r
            l_start = self.start_joints_l
        else:
            r_start = start_joints['right']
            l_start = start_joints['left']

        # plan cartesian path
        valid = False
        try:
            self.mp_func(
                tip_right,
                tip_left,
                force_start=l_start+r_start,
                avoid_collisions=avoid_collisions,
                jump_thresh=jump_thresh
            )
            valid = True
        except PlanWaypointsError:
            pass

        return valid

    def mp_func(self, tip_right, tip_left, force_start, avoid_collisions=None, jump_thresh=None):
        if avoid_collisions is None:
            avoid_collisions = self.avoid_collisions
        self.robot.mp_right.plan_waypoints(
            tip_right,
            force_start=force_start,
            avoid_collisions=avoid_collisions,
            jump_thresh=jump_thresh
        )


class PushLeftSkill(PushRightSkill):
    def __init__(self, sampler, robot, get_plan_func, ignore_mp=False, avoid_collisions=True):
        super(PushLeftSkill, self).__init__(sampler, robot, get_plan_func, ignore_mp, avoid_collisions)
        self.skill_name = 'push_left'

    def sample(self, state, *args, **kwargs):
        final_trans = False
        if 'final_trans' in kwargs.keys():
            final_trans = kwargs['final_trans']
        if final_trans:
            final_trans_to_go = state.transformation_to_go
        else:
            final_trans_to_go = None

        pcd_pts = copy.deepcopy(state.pointcloud)
        pcd_pts[:, 1] = -pcd_pts[:, 1]
        pcd_pts_full = None
        if state.pointcloud_full is not None:
            pcd_pts_full = copy.deepcopy(state.pointcloud_full)
            pcd_pts_full[:, 1] = -pcd_pts_full[:, 1]

        prediction = self.sampler.sample(
            pcd_pts,
            state_full=pcd_pts_full,
            final_trans_to_go=final_trans_to_go)

        if final_trans:
            # on last step we have to trust that this is correct
            new_transformation = prediction['transformation']
        else:
            # if in the middle, then flip based on the right pull transform
            new_transformation = copy.deepcopy(prediction['transformation'])
            new_transformation[0, 1] *= -1
            new_transformation[1, 0] *= -1
            new_transformation[1, -1] *= -1
        new_palms = util.pose_stamped2np(util.flip_palm_pulling(util.list2pose_stamped(prediction['palms'][:7])))
        new_palms[1] *= -1

        new_state = PointCloudNode()
        new_state.init_state(state, new_transformation, skill=self.skill_name)
        # new_state.init_palms(new_palms)
        new_state.init_palms(new_palms,
                             correction=True,
                             prev_pointcloud=state.pointcloud_full,
                             dual=False)
        return new_state

    def mp_func(self, tip_right, tip_left, force_start, avoid_collisions=None, jump_thresh=None):
        if avoid_collisions is None:
            avoid_collisions = self.avoid_collisions
        self.robot.mp_left.plan_waypoints(
            tip_left,
            force_start=force_start,
            avoid_collisions=avoid_collisions,
            jump_thresh=jump_thresh
        )
