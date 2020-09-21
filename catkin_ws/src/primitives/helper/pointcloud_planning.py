import os, sys
import os.path as osp
import pickle
import numpy as np
import random

import trimesh
import open3d
import pcl
import pybullet as p

import copy
import time
from IPython import embed

from yacs.config import CfgNode as CN
from airobot.utils import common

sys.path.append('/root/catkin_ws/src/primitives/')
# from helper import util2 as util
# from helper import registration as reg
import util2 as util
import registration as reg
from closed_loop_experiments_cfg import get_cfg_defaults
from eval_utils.visualization_tools import correct_grasp_pos, project_point2plane
from pointcloud_planning_utils import (
    PointCloudNode, PointCloudCollisionChecker, PointCloudPlaneSegmentation,
    PalmPoseCollisionChecker, StateValidity, PlanningFailureModeTracker)


class PointCloudTree(object):
    """RRT-style planning search tree, using nodes that contain
    information about pointclouds, robot end-effector contact poses,
    and relative transformations from parents.

    Args:
        start_pcd (np.ndarray): N X 3 array of points, [x, y, z], representing
            the initial state of the object
        trans_des (np.ndarray): 4 X 4 homogenous transformation matrix, describing
            the overall change in pose that the object should undergo
        skeleton (list): Specifies what skill sequence should be used to reach the goal
        skills (dict): Dictionary with references to the individual skills
        start_pcd_full (np.ndarray, optional): N X 3 array of points, [x, y, z].
            Different than `start_pcd` because this is the full pointcloud (not 
            downsampled).
        motion_planning (bool, optional): Whether or not motion planning should be performed.
            Defaults to True.
        only_rot (bool, optional): Whether or not to only check the orientation component
            of the transformation reached so far. Defaults to True.
        target_surfaces (list, optional): List, corresponding to each step in the skeleton,
            of what target surfaces to use for sampling subgoals. Defaults to None.
        visualize (bool, optional): Whether or not to use PyBullet to visualize planning progress. 
            Defaults to False.
        obj_id (int, optional): PyBullet object id, for visualization. Defaults to None.
        start_pose (PoseStamped, optional): Initial pose of the object in the environment,
            for visualizing planning progress. Defaults to None.
        collision_pcds (list, optional): List of pointclouds (N X 3 np.ndarray)in the environment 
            that should be considered as obstacles and used for collision checking between objects
            during planning. Defaults to None.

    Attributes:
        TODO
    """
    def __init__(self, start_pcd, trans_des, skeleton, skills, max_steps,
                 start_pcd_full=None, motion_planning=True,
                 only_rot=True, target_surfaces=None,
                 visualize=False, obj_id=None, start_pose=None,
                 collision_pcds=None, start_goal_palm_check=False, tracking_failures=False):
        self.skeleton = skeleton
        self.skills = skills
        self.goal_threshold = None
        self.motion_planning = motion_planning
        self.timeout = 300
        self.only_rot = only_rot
        self.start_goal_palm_check = start_goal_palm_check
        self.tracking_failures = tracking_failures

        self.pos_thresh = 0.005
        self.ori_thresh = np.deg2rad(15)
        self.eye_thresh = 1.0
        self.k_max = 10
        self.final_step_probability = 0.5
        self.max_plan_steps = max_steps

        self.buffers = {}

        for i in range(len(skeleton)):
            self.buffers[i+1] = []
        # for i in range(max_steps):
        #     self.buffers[i+1] = []

        if target_surfaces is None:
            self.target_surfaces = [None]*len(skeleton)
        else:
            self.target_surfaces = target_surfaces

        self.visualize = False
        self.object_id = None
        self.start_pose = None
        if visualize and obj_id is not None and start_pose is not None:
            self.visualize = True
            self.object_id = obj_id
            self.start_pose = start_pose

        self.collision_pcds = collision_pcds
        self.pcd_collision_checker = PointCloudCollisionChecker(self.collision_pcds)
        self.palm_collision_checker = PalmPoseCollisionChecker()
        self.pcd_segmenter = PointCloudPlaneSegmentation()

        self.planning_stat_tracker = PlanningFailureModeTracker(skeleton)

        # initialize the start node of the planning tree
        self.initialize_pcd(start_pcd, start_pcd_full, trans_des)

    def initialize_pcd(self, start_pcd, start_pcd_full, trans_des):
        """Function to setup the initial point cloud node with all it's necessary resources
        for efficient planning. This includes segmenting all the planes in the point cloud,
        estimating the point cloud normals, pairing planes that are likely to be antipodal 
        w.r.t. each other, and tracking their average normal directions.
        """
        # initialize the basics
        self.start_node = PointCloudNode()
        self.start_node.set_pointcloud(start_pcd, start_pcd_full)
        self.start_node.set_trans_to_go(trans_des)
        self.transformation = np.eye(4)

        pointcloud_pts = start_pcd if start_pcd_full is None else start_pcd_full

        # perform plane segmentation
        planes = self.pcd_segmenter.get_pointcloud_planes(pointcloud_pts)

        # save planes in start node so they can be tracked
        self.start_node.set_planes(planes)

        self.buffers[0] = [self.start_node]

    def plan(self):
        """RRT-style sampling-based planning loop, that assumes a plan skeleton is given. 
        General logic follows: 
        1. stepping through plan skeleton, 
        2. using skill samplers and node buffers to sample "start" states and possible 
           actions/subgoals from those start states
        3. checking feasibility based on collisions, motion planning, and other heuristics
        4. seeing if we can eventually reach the goal state, and returning a plan

        Returns:
            list: Plan of PointCloudNode instances that moves the initial pointcloud into
                the goal pointcloud through a sequence of rigid transformations
        """
        done = False
        start_time = time.time()
        while not done:
            for i, skill in enumerate(self.skeleton):
                if i < len(self.skeleton) - 1:
                    k = 0
                    while True:
                        # track total number of samples
                        self.planning_stat_tracker.increment_total_counts(skill)

                        if time.time() - start_time > self.timeout:
                            print('Timed out')
                            return None
                        k += 1
                        if k > self.k_max:
                            valid = False
                            break
                        sample, index = self.sample_next(i, skill)
                        if sample is None:
                            break

                        if self.visualize:
                            sample_pose = util.transform_pose(self.start_pose, util.pose_from_matrix(sample.transformation_so_far))
                            sample_pose_np = util.pose_stamped2np(sample_pose)
                            p.resetBasePositionAndOrientation(self.object_id, sample_pose_np[:3], sample_pose_np[3:])

                        # check if this satisfies the constraints of the next skill
                        valid_preconditions = self.skills[self.skeleton[i+1]].satisfies_preconditions(sample)
                        valid = valid_preconditions

                        tic = time.time()
                        start_palm_collision, goal_palm_collision = self.palm_collision_checker.sample_in_start_goal_collision(sample)
                        sg_palm_coll_t = time.time() - tic

                        valid = valid and (not start_palm_collision and not goal_palm_collision)

                        # perform 2D pointcloud collision checking, for other objects
                        # valid = valid and self.pcd_collision_checker.check_2d(sample.pointcloud)

                        # if we are tracking failures, run motion feasibility check even if we're not valid.
                        # otherwise, only check feasibility if previous checks passed
                        check_motion_feasibility = True if self.tracking_failures else valid

                        # check if this is a valid transition (motion planning)
                        if check_motion_feasibility:
                            if self.motion_planning:
                                tic = time.time()
                                feasible_motion = self.skills[skill].feasible_motion(sample)
                                feasible_motion_t = time.time() - tic
                                valid = valid and feasible_motion

                        if valid:
                            sample.parent = (i, index)
                            self.buffers[i+1].append(sample)
                            break

                        # record what happened to us if things weren't valid
                        self.planning_stat_tracker.update_infeasibility_counts(
                            (not valid_preconditions, 0.0, skill),
                            (start_palm_collision, sg_palm_coll_t, skill),
                            (goal_palm_collision, sg_palm_coll_t, skill),
                            (not feasible_motion, feasible_motion_t, skill)
                        )

                        time.sleep(0.01)
                else:
                    # sample is the proposed end state, which has the path encoded
                    # via all its parents
                    # sample, index = self.sample_next(i, skill)
                    sample, index = self.sample_final(i, skill)
                    if sample is None:
                        continue
                    if self.visualize:
                        sample_pose = util.transform_pose(self.start_pose, util.pose_from_matrix(sample.transformation_so_far))
                        sample_pose_np = util.pose_stamped2np(sample_pose)
                        p.resetBasePositionAndOrientation(self.object_id, sample_pose_np[:3], sample_pose_np[3:])

                    if not self.skills[skill].valid_transformation(sample):
                        # pop sample that this came from
                        self.buffers[i].pop(index)
                        continue

                    # still check motion planning for final step
                    print('final mp')
                    if self.motion_planning:
                        valid = self.skills[skill].feasible_motion(sample)
                    else:
                        valid = True

                    if sample is not None and valid:
                        sample.parent = (i, index)
                        reached_goal = self.reached_goal(sample)
                        if reached_goal:
                            done = True
                            self.buffers['final'] = sample
                            break
                    else:
                        pass
            if time.time() - start_time > self.timeout:
                print('Timed out')
                return None

        # extract plan
        plan = self.extract_plan()
        return plan

    def plan_max_length(self):
        """RRT-style sampling-based planning loop, that assumes a plan skeleton is not given,
        but that a maximum plan length is given. 
        General logic follows: 
        1. stepping toward the maximum number of skeleton steps 
        2. popping states from node buffers, checking skill preconditions, and sampling
           skills to be used for that steps in the skeleton.
        3. using skill samplers with sampled "start" states and possible 
           actions/subgoals from those start states, tracking which skill is used on which step
        4. checking feasibility based on collisions, motion planning, and other heuristics
        5. seeing if we can eventually reach the goal state, and returning a plan

        Returns:
            list: Plan of PointCloudNode instances that moves the initial pointcloud into
                the goal pointcloud through a sequence of rigid transformations
        """        
        done = False
        start_time = time.time()
        while not done:
            for plan_step in range(self.max_plan_steps):
                k = 0
                while True:
                    if time.time() - start_time > self.timeout:
                        print('Timed out')
                        return None 
                    k += 1
                    if k > self.k_max:
                        break

                    # option 1: sample start state, check which preconditions are satisfied,
                    # and then sample only from those skills

                    # option 2: sample skill, then sample start state, then check precondition
                    # and proceed. if not valid, then start over from new skill, or try a different
                    # skill with this particular start state that was found

                    # sample skill from potential skills to use and check validity
                    valid_skill = False
                    valid_skill_sample = 0
                    start_sample, index = self.sample_buffer(plan_step)
                    if start_sample is None:
                        # no samples in the corresponding buffer, keep going at next step
                        break

                    while not valid_skill:
                        # try to find a skill that works with whatever start state we have popped
                        skill = self.sample_skill()
                        valid_skill = self.skills[skill].satisfies_preconditions(start_sample)
                        valid_skill_sample += 1
                        if valid_skill or (valid_skill_sample > len(self.skills) - 1):
                            break

                    if not valid_skill:
                        print('did not find valid skill')
                        break

                    print('Skill: ' + str(skill))
                    # check if we have reached final step, or use specified probability to sometimes sample to the goal
                    final_step = (np.random.random() > self.final_step_probability) or (plan_step == self.max_plan_steps - 1)
                    sample = self.sample_from_skill(skill, start_sample, final_step=final_step)

                    if self.visualize:
                        sample_pose = util.transform_pose(self.start_pose, util.pose_from_matrix(sample.transformation_so_far))
                        sample_pose_np = util.pose_stamped2np(sample_pose)
                        p.resetBasePositionAndOrientation(self.object_id, sample_pose_np[:3], sample_pose_np[3:])

                    # check validity of the transition
                    if self.motion_planning:
                        valid = self.skills[skill].feasible_motion(sample)
                    else:
                        valid = True
                
                    if valid:
                        sample.parent = (plan_step, index)
                        sample.skill = skill
                        if final_step:
                            # check if we have found the goal
                            reached_goal = self.reached_goal(sample)
                            if reached_goal:
                                done = True
                                self.buffers['final'] = sample
                        else:
                            # add to buffer for future popping
                            self.buffers[plan_step+1].append(sample)
                        break
                    else:
                        # try again, until we reach self.k_max
                        continue
        # extract plan
        plan = self.extract_plan()
        return plan

    def sample_skill(self):
        """Sample a skill to use for a particular step in the unknown skeleton

        Returns:
            str: Name of skill to use        
        """
        skill = random.sample(self.skills.keys(), 1)[0]
        return skill

    def sample_buffer(self, i):
        """Pop a starting state from the buffer, depending on which step in the
        skeleton we are on

        Args:
            i (int): Index of the current step being sampled in the skeleton

        Returns:
            2-element tuple containing:
            - PointCloudNode: Sample popped from corresponding buffer of nodes that have been saved
            - int: Index of the popped buffer sample
        """
        buffer_state, index = None, None
        if i == 0:
            buffer_state = self.start_node
            index = 0
        else:
            if len(self.buffers[i]) > 0:
                index = np.random.randint(len(self.buffers[i]))
                buffer_state = self.buffers[i][index]
        return buffer_state, index

    def sample_from_skill(self, skill, start_sample, final_step=False):
        """Given a start state and specific skill to use, obtain a sample from
        the specified skill sampler

        Args:
            skill (str): Which skill to sample from
            start_sample (PointCloudNode): Which start state to use for skill sampling
            index (int): Index of the start state in the corresponding plan skeleton buffer

        Returns:
            PointCloudNode: Sample of next node, whose feasibility will be checked
        """
        target_surface = random.sample(self.target_surfaces, 1)[0]
        sample = self.skills[skill].sample(
            start_sample,
            target_surface=target_surface,
            final_trans=final_step
        )
        return sample

    def sample_next(self, i, skill):
        """Sample a PointCloudNode using the skill samplers and some
        initial node that has already been saved in the plan buffers

        Args:
            i (int): Index of the step in the plan skeleton currently being sampled
            skill (str): Name of the skill to be used for sampling the next node.
                string is used as the key in the internal dictionary of references
                to the skills. (skill = self.skills.keys()[i])

        Returns:
            2-element tuple containing:
            - PointCloudNode: Sample of next node, whose feasibility will be checked
            - int: Buffer index of the initial node used to sample from the skill
        """
        sample, index = None, None
        last_step = i == len(self.skeleton) - 1
        if i == 0:
            # sample from first skill if starting at beginning
            sample = self.skills[skill].sample(
                self.start_node,
                target_surface=self.target_surfaces[i],
                final_trans=last_step)
            index = 0
        else:
            # sample from the buffers we have
            if len(self.buffers[i]) > 0:
                index = np.random.randint(len(self.buffers[i]))
                state = self.buffers[i][index]
                sample = self.skills[skill].sample(
                    state,
                    target_surface=self.target_surfaces[i],
                    final_trans=last_step)
        return sample, index

    def sample_final(self, i, skill):
        """Sample a PointCloudNode using the skill samplers and some
        initial node that has already been saved in the plan buffers. Separate
        method used to sample final skill so that we are sure that the computed
        goal transformation is used instead of the random subgoal. 

        Args:
            i (int): Index of the step in the plan skeleton currently being sampled
            skill (str): Name of the skill to be used for sampling the next node.
                string is used as the key in the internal dictionary of references
                to the skills. (skill = self.skills.keys()[i])

        Returns:
            2-element tuple containing:
            - PointCloudNode: Sample of next node, whose feasibility will be checked
            - int: Buffer index of the initial node used to sample from the skill
        """
        sample, index = None, None
        if len(self.buffers[i]) > 0:
            index = np.random.randint(len(self.buffers[i]))
            state = self.buffers[i][index]
            sample = self.skills[skill].sample(
                state,
                final_trans=True)
        return sample, index

    def reached_goal(self, sample):
        """Checks to see if a particular PointCloudNode satisfies the task-specification.
        This is done by checking if the sequence of transformations leading to this state,
        represented by the tracked `transformation_so_far` attribute of the node sample, is
        within a position and orientation threshold of the desired transformation that specifies
        the task

        Args:
            sample (PointCloudNode): Node in the search tree that is being checked, to see if
                it is within a threshold of the goal node.

        Returns:
            bool: True if the sample is within a threshold of satisfying the goal specification
        """
        T_eye = np.eye(4)
        T_so_far = sample.transformation_so_far
        T_to_go = sample.transformation_to_go

        T_so_far_pose = util.pose_stamped2np(util.pose_from_matrix(T_so_far))
        T_des_pose = util.pose_stamped2np(util.pose_from_matrix(self.start_node.transformation_to_go))

        pos_err, ori_err = util.pose_difference_np(T_so_far_pose, T_des_pose)
        eye_diff_1 = T_to_go[:-1, :-1] - T_eye[:-1, :-1]
        eye_diff_2 = T_to_go - T_eye
        print('pos err: ' + str(pos_err) +
              ' ori err: ' + str(ori_err) +
              ' eye norm 1: ' + str(np.linalg.norm(eye_diff_1)) +
              ' eye norm 2: ' + str(np.linalg.norm(eye_diff_2)))

        if self.only_rot:
            eye_diff = T_to_go[:-1, :-1] - T_eye[:-1, :-1]
            return ori_err < self.ori_thresh
        else:
            eye_diff = T_to_go - T_eye
            return ori_err < self.ori_thresh and pos_err < self.pos_thresh
        return np.linalg.norm(eye_diff) < self.eye_thresh

    def extract_plan(self):
        """Backtrack through plan from final node reached using parent nodes,
        until start node is reached, and return as plan to be follosed

        Returns:
            list: Sequence of PointCloudNodes that connect start node and goal node
        """
        node = self.buffers['final']
        parent = node.parent
        plan = []
        plan.append(node)
        while parent is not None:
            node = self.buffers[parent[0]][parent[1]]
            plan.append(node)
            parent = node.parent
        plan.reverse()
        return plan
