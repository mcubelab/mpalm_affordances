import os.path as osp
import sys
import time
import lcm

import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
from rpo_lcm import string_array_t 
from rpo_planning.utils import common as util
from rpo_planning.utils import lcm_utils

class SkeletonSampler:
    """Class to interface with neural network processes running that will
    take an environment observation and task specification as input
    and return a sequence of high-level skill actions to take as output.
    Uses LCM interprocesses communication
    """
    def __init__(self, pcd_pub_name='explore_pcd_obs', task_pub_name='explore_task_obs',
                 sub_name='explore_skill_skeleton'):
        self.pcd_pub_name = pcd_pub_name 
        self.task_pub_name = task_pub_name
        self.sub_msg_name = sub_name
        self.lc = lcm.LCM()
        self.subscription = self.lc.subscribe(self.sub_msg_name, self.sub_handler)
        self.model_path = None
        self.samples_count = 0

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

        self.lc.publish(self.pcd_pub_name, pcd_msg.encode())
        self.lc.publish(self.task_pub_name, task_msg.encode())

        self.received_data = False
        while not self.received_data:
            self.lc.handle()

        predicted_skeleton = [str(skill) for skill in self.skeleton]
        return predicted_skeleton
        
    def sub_handler(self, channel, data):
        """
        Callback for handling incoming data from the LCM publisher on the other end.
        Data should contain the high-level skill/skeleton predictions.

        Args:
            channel ([type]): [description]
            data ([type]): [description]
        """
        msg = string_array_t.decode(data)
        self.skeleton = msg.string_array 
        self.received_data = True

if __name__ == "__main__":
    import numpy as np
    pts = np.random.rand(100, 3)
    task = util.matrix_from_pose(util.unit_pose())
    skeleton_sampler = SkeletonSampler()
    
    skeleton = skeleton_sampler.predict(pts, task)

    print(' Got skeleton: ', skeleton)
    from IPython import embed
    embed()

