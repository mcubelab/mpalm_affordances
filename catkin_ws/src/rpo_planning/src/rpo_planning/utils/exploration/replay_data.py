import os.path as osp
import sys
from collections import namedtuple

import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
from rpo_lcm import rpo_transition_t, rpo_plan_t, dual_pose_stamped_t 
from rpo_planning.utils import lcm_utils

RPOTransition = namedtuple(
    'RPOTransition',
    [
        'node',
        'observation',
        'action',
        'action_index',
        'reward',
        'achieved_goal',
        'desired_goal',
        'done'
    ])

def rpo_plan2lcm(plan):
    """
    Function to convert the raw sequence of transitions
    into a message to be send via LCM

    Args:
        plan (list): Each element is an RPOTransition containing the info
            to send over

    Returns:
        rpo_plan_t: LCM message to be sent over
    """
    lcm_plan = rpo_plan_t()
    lcm_plan.num_steps = len(plan)

    for i, transition in enumerate(plan):
        rpo_t = rpo_transition_t()

        # fill in general plan stuff
        rpo_t.observation = lcm_utils.np2point_cloud_t(transition.observation)
        rpo_t.action = transition.action
        rpo_t.action_index = transition.action_index
        rpo_t.reward = transition.reward
        rpo_t.achieved_goal = lcm_utils.matrix2pose_stamped_lcm(transition.achieved_goal)
        rpo_t.desired_goal = lcm_utils.matrix2pose_stamped_lcm(transition.desired_goal)
        rpo_t.done = transition.done

        # fill in specific skill parameters as well 
        rpo_t.skill_parameters.num_points = transition.observation.shape[0]
        pcd_mask = transition.node.pointcloud_mask
        if pcd_mask is not None:    
            rpo_t.skill_parameters.mask_probs = pcd_mask
        else:
            rpo_t.skill_parameters.mask_probs = [0.0] * rpo_t.skill_parameters.num_points 
        lcm_utils.fill_skill_subgoal_and_contact(
            contact_pose=transition.node.palms,
            subgoal_pose=transition.node.transformation,
            sp_t=rpo_t.skill_parameters
        )

        lcm_plan.plan.append(rpo_t)
    return lcm_plan
