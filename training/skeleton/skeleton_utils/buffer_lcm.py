import os.path as osp
import sys
import lcm
import numpy as np

import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
from rpo_lcm import rpo_transition_t, rpo_plan_t, dual_pose_stamped_t 
from rpo_planning.utils import lcm_utils

from skeleton_utils.skeleton_globals import SOS_token, EOS_token, PAD_token


class BufferLCM:
    def __init__(self, replay_buffer, in_msg_name='rpo_transition_sequences'):
        # LCM message sub names for transition data coming in
        self.in_msg_name = in_msg_name

        self.lc = lcm.LCM()
        self.sub = self.lc.subscribe(self.in_msg_name, self.sub_handler)

        # structure containing all the transitions that model is training on
        self.replay_buffer = replay_buffer
    
    def sub_handler(self, channel, data):
        """
        Callback function to receive LCM data containing the sequence of 
        RPO transitions

        Args:
            channel ([type]): [description]
            data ([type]): [description]
        """
        msg = rpo_plan_t.decode(data)

        self.msg = msg
        self.received_transition_data = True

    def receive_and_append_buffer(self):
        """
        Main loop to wait for messages to come in from LCM containing
        RPO transitions from exploration procedure that should then
        populate the replay buffer on the model side 
        """

        self.received_transition_data = False
        
        while True:
            self.lc.handle()
            if self.received_transition_data:
                break

        print('got message!')
        # unpack message that came in and format for appending replay buffer
        msg = self.msg

        # decode RPO transition sequence
        for i, transition in enumerate(msg.plan):
            # get observation
            o = lcm_utils.unpack_pointcloud_lcm(
                transition.observation.points,
                transition.observation.num_points)
            
            # get action
            a = transition.action_index
            
            # TODO: get next observation
            o_ = o

            # get reward
            r = transition.reward

            # get goal info
            ach_goal = lcm_utils.pose_stamped2list(transition.achieved_goal)
            des_goal = lcm_utils.pose_stamped2list(transition.desired_goal)
            done_flag = bool(transition.done)

            # append replay buffer
            self.replay_buffer.append(
                observation=np.asarray(o), 
                action=a, 
                next_observation=np.asarray(o_), 
                reward=r, 
                done=done_flag, 
                achieved_goal=np.asarray(ach_goal),
                desired_goal=np.asarray(des_goal))

            # get skill param info
            mask = transition.skill_parameters.mask_probs
            tp = lcm_utils.pose_stamped2list(transition.skill_parameters.subgoal_pose)
            # just take the contact associated with the first point (assume they're all the same)
            tc_r = lcm_utils.pose_stamped2list(transition.skill_parameters.contact_pose[0].right_pose) 
            tc_l = lcm_utils.pose_stamped2list(transition.skill_parameters.contact_pose[0].left_pose) 
            tc = tc_r + tc_l

            # applend replay buffer with skill params
            self.replay_buffer.append_skill_params(
                contact=np.asarray(tc),
                subgoal=np.asarray(tp),
                mask=np.asarray(mask)
            )
        print('appended!')
        return True

if __name__ == "__main__":
    from replay_buffer import TransitionBuffer
    import torch
    buffer = TransitionBuffer(
        size=5000,
        observation_n=(100, 3),
        action_n=1,
        device=torch.device('cuda:0'),
        goal_n=7)
        

    buffer_to_lcm = BufferLCM(buffer)

    try:
        while True:
            buffer_to_lcm.receive_and_append_buffer()
            if buffer.index > 100: 
                break
    except KeyboardInterrupt:
        pass

    from IPython import embed
    embed()
    # observations, actions, next_observations, rewards, not_dones, des_goals, ach_goals = buffer.sample(n=10, sequence_length=5)
    subgoal, contact, observation, next_observation, action_seq = buffer.sample_sg(n=10) 
    print('done')
