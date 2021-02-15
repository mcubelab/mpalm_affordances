import time
import pybullet as p
import numpy as np

import rpo_planning.utils.common as util
from rpo_planning.utils.exploration.replay_data import RPOTransition
from rpo_planning.pointcloud_planning.rpo_planner import PointCloudTree

class PointCloudTreeLearner(PointCloudTree):
    """RRT-style planning search tree, using nodes that contain
    information about pointclouds, robot end-effector contact poses,
    and relative transformations from parents.

    Args:
        start_pcd (np.ndarray): N X 3 array of points, [x, y, z], representing
            the initial state of the object
        trans_des (np.ndarray): 4 X 4 homogenous transformation matrix, describing
            the overall change in pose that the object should undergo
        skeleton (list): Specifies what skill sequence should be used to reach the goal
        skeleton_indices (list): Categorical indices that correspond to each skill in the
            skeleton, for encoding among the full skill language on the neural network side
        skills (dict): Dictionary with references to the individual skills
        start_pcd_full (np.ndarray, optional): N X 3 array of points, [x, y, z].
            Different than `start_pcd` because this is the full pointcloud (not
            downsampled).
        max_steps (int): Maximum number of skills to use in the plan skeleton, if it's not
            provided.
        skeleton_policy (rpo_planning.exploration.skeleton_sampler.SkeletonSampler): 
            Interface to the policy that is used to select high-level skills
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
    def __init__(self, start_pcd, trans_des, skeleton, skeleton_indices, skills, max_steps, skeleton_policy,
                 start_pcd_full=None, motion_planning=True,
                 only_rot=True, target_surfaces=None,
                 visualize=False, obj_id=None, start_pose=None,
                 collision_pcds=None, start_goal_palm_check=False, tracking_failures=False,
                 max_relabel_samples=50):
        super(PointCloudTreeLearner, self).__init__(
                 start_pcd, trans_des, skeleton, skills, max_steps,
                 start_pcd_full=start_pcd_full, motion_planning=motion_planning,
                 only_rot=only_rot, target_surfaces=target_surfaces,
                 visualize=visualize, obj_id=obj_id, start_pose=start_pose,
                 collision_pcds=collision_pcds, start_goal_palm_check=start_goal_palm_check, tracking_failures=tracking_failures)

        self.skeleton_indices = skeleton_indices
        self._make_skill_lang()
        self.skeleton_policy = skeleton_policy
        self.max_relabel_samples = max_relabel_samples

    def _make_skill_lang(self):
        """
        Function to create a mapping between the skill names and the indices encoding them for the NN 
        """
        self.skill2index, self.index2skill = {}, {}
        for i, skill in enumerate(self.skeleton):
            self.index2skill[i] = skill
            self.skill2index[skill] = i 

    def plan_with_skeleton(self):
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
        plan = self.plan()
        # process plan to get reward information

        return plan

    def plan_without_skeleton(self):
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
                        skill = self.sample_skill(start_sample)
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

    def sample_skill(self, sample):
        """Sample a skill type using the policy, taking the current state as input

        Args:
            sample (PointCloudNode): State to use when sampling skill type (action)
        """
        skill = self.skeleton_policy.predict(sample.pointcloud, self.start_node.transformation_to_go)
        return skill

    def compute_achieved_goal(self):
        # search through leaf nodes of the buffer to find a state that is close to the goal we were trying to reach
        k_sampled = 0

        # non-zero buffers
        non_empty_buffer_idxs = []
        for i in range(len(self.buffers)):
            if len(self.buffers[i]) > 0:
                non_empty_buffer_idxs.append(i)
        non_empty_buffer_idxs = sorted(non_empty_buffer_idxs)

        best_buffer, best_idx = None, None
        while True:
            # sample from last buffer
            buffer_idx = np.random.choice(non_empty_buffer_idxs, 1)
            idx = np.random.randint(len(self.buffers[buffer_idx]))
            sample = self.buffers[-1][idx]
            # sample = random.sample(self.buffers[-1], 1)[0]

            T_eye = np.eye(4)
            T_so_far = sample.transformation_so_far
            T_to_go = sample.transformation_to_go

            T_so_far_pose = util.pose_stamped2np(util.pose_from_matrix(T_so_far))
            T_des_pose = util.pose_stamped2np(util.pose_from_matrix(self.start_node.transformation_to_go))

            pos_err, ori_err = util.pose_difference_np(T_so_far_pose, T_des_pose)
            if best_buffer is None or best_idx is None:
                best_buffer, best_idx = buffer_idx, idx
                best_pos_err, best_ori_err = pos_err, ori_err

            if pos_err < best_pos_err or ori_err < best_ori_err:
                best_buffer, best_idx = buffer_idx, idx
                best_pos_err, best_ori_err = pos_err, ori_err

            k_sampled += 1
            if k_sampled > self.max_relabel_samples:
                break

        # set this sample which we have obtained from the buffer as the final sample
        self.buffers['final'] = sample
        plan = self.extract_plan()
        return plan

    def process_plan_transitions(self, plan):
        # compute achieved goal, for hindsight relabeling
        if plan is None:
            plan = self.compute_achieved_goal()

        # put reward and transition info into plan
        processed_plan = []
        for t, node in enumerate(plan):
            data = {}
            data['node'] = node
            data['observation'] = node.pointcloud
            data['action'] = node.skill
            data['action_index'] = self.skill2index[node.skill]
            data['reward'] = -1 if t < len(plan) - 1 else 0 
            # data['reward'] = 0 if t < len(plan) else 1
            data['achieved_goal'] = plan[-1].transformation_so_far
            data['desired_goal'] = self.start_node.transformation_to_go
            data['done'] = False if t < len(plan) - 1 else True
            processed_plan.append(RPOTransition(**data))

        return processed_plan