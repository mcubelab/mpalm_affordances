import os.path as osp
import sys
import time
import lcm

from airobot import set_log_level, log_debug, log_info, log_warn, log_critical

import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
from rpo_lcm import string_array_t, rpo_plan_skeleton_t 
from rpo_planning.utils import common as util
from rpo_planning.utils import lcm_utils

class SkeletonSampler:
    """Class to interface with neural network processes running that will
    take an environment observation and task specification as input
    and return a sequence of high-level skill actions to take as output.
    Uses LCM interprocesses communication
    """
    def __init__(self, pcd_pub_name='explore_pcd_obs', task_pub_name='explore_task_obs',
                 sub_name='explore_skill_skeleton', rpo_plan_sub_name='rpo_transition_sequences',
                 timeout=10.0, verbose=False):
        # name of pub/sub messages containing the task description (point cloud + desired trans)
        # and skeleton prediction from model
        self.pcd_pub_name = pcd_pub_name 
        self.task_pub_name = task_pub_name
        self.sub_msg_name = sub_name

        # name of LCM message that is sent, containing the transitions to be added to replay buffer
        self.transition_pub_name = rpo_plan_sub_name

        self.lc = lcm.LCM()
        self.subscription = self.lc.subscribe(self.sub_msg_name, self.sub_handler)
        self.model_path = None
        self.samples_count = 0
        self.timeout = timeout
        self.verbose = verbose

    def predict(self, pointcloud, transformation_des):
        """Function to take the observation, in the form of a point cloud,
        along with the task specification, in the form of a desired
        rigid body transformation of the point cloud, and sent it to the 
        neural network process. Returns whatever skill sequence message
        that the neural network predicts

        Args:
            pointcloud (np.ndarray): [N x 3] array of [x, y, z] points
            transformation_des (np.ndarray): [4 x 4] transformation matrix

        Returns:
            list: Sequence of skill types (each element will be a string)
        """
        self.samples_count += 1

        pcd_msg = lcm_utils.np2point_cloud_t(pointcloud)
        pcd_msg.header.seq = self.samples_count
        pcd_msg.header.utime = time.time() / 1e6

        task_msg = lcm_utils.matrix2pose_stamped_lcm(transformation_des)

        while True:
            if self.verbose:
                log_debug('Skeleton sampler: publishing task encoding')
            self.lc.publish(self.pcd_pub_name, pcd_msg.encode())
            self.lc.publish(self.task_pub_name, task_msg.encode())

            self.received_data = False
            self.lc.handle_timeout(self.timeout*1e3)
            if self.verbose:
                log_debug('Skeleton sampler: handler time out after %.1fs, looping once more' % self.timeout)
            if self.received_data:
                break

        predicted_skeleton = [str(skill) for skill in self.skeleton]
        predicted_skeleton_inds = [int(ind) for ind in self.skeleton_inds]
        return predicted_skeleton, predicted_skeleton_inds
        
    def sub_handler(self, channel, data):
        """
        Callback for handling incoming data from the LCM publisher on the other end.
        Data should contain the high-level skill/skeleton predictions.

        Args:
            channel ([type]): [description]
            data ([type]): [description]
        """
        msg = rpo_plan_skeleton_t.decode(data)
        self.skeleton = msg.skill_names.string_array 
        self.skeleton_inds = msg.skill_indices
        if self.verbose:
            log_debug('Skeleton sampler sub handler: %s' % ', '.join(self.skeleton))
        self.received_data = True

    def add_to_replay_buffer(self, processed_plan):
        """
        Function to send data over to the replay buffer via LCM

        Args:
            processed_plan (rpo_planning.lcm_types.rpo_plan_t): Sequence of RPO transitions
        """
        self.lc.publish(self.transition_pub_name, processed_plan.encode())


if __name__ == "__main__":
    import numpy as np
    pts = np.random.rand(100, 3)
    task = util.matrix_from_pose(util.unit_pose())
    skeleton_sampler = SkeletonSampler()
    
    skeleton = skeleton_sampler.predict(pts, task)

    print(' Got skeleton: ', skeleton)
    from IPython import embed
    embed()

