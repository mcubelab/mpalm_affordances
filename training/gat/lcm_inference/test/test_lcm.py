import os, os.path as osp
import sys
import numpy as np
import lcm
import time

import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
from rpo_lcm import point_t, quaternion_t, pose_stamped_t, pose_t, point_cloud_t, skill_param_t, skill_param_array_t
from rpo_planning.utils import common as util
from rpo_planning.utils import lcm_utils


class PredictorBaseLCM(object):
    def __init__(self, pub_msg_name, sub_msg_name):
        self.pub_msg_name = pub_msg_name
        self.sub_msg_name = sub_msg_name
        self.lc = lcm.LCM()
        self.lcm_subscription = self.lc.subscribe(self.sub_msg_name, self.sub_handler)

    def sub_handler(self, channel, data):
        msg = point_cloud_t.decode(data)
        points = msg.points

        print(' Points: ', points)

        # self.points = points
        self.points = []
        num_pts = msg.num_points
        for i in range(num_pts):
            pt = [
                points[i].x,
                points[i].y,
                points[i].z]
            self.points.append(pt)
        self.received_data = True

    def infer_single(self):
        self.received_data = False
        while not self.received_data:
            self.lc.handle()
        
        points = self.points

        print( 'Got points: ', points)
        pub_msg = skill_param_t()
        pub_msg.num_points = 100
        for _ in range(pub_msg.num_points):
            pub_msg.contact_pose.append(lcm_utils.list2pose_stamped_lcm([0, 0, 0, 0, 0, 0, 1]))
        pub_msg.subgoal_pose = lcm_utils.list2pose_stamped_lcm([1, 1, 1, 1, 0, 0, 0])
        pub_msg.mask_probs = np.random.rand(pub_msg.num_points).tolist()

        self.lc.publish(self.pub_msg_name, pub_msg.encode())

    def infer_array(self):
        self.received_data = False
        while not self.received_data:
            self.lc.handle()
        
        points = self.points

        print( 'Got points: ', points)
        pub_msg = skill_param_array_t()
        pub_msg.num_entries = 2
        for _ in range(pub_msg.num_entries):
            skp = skill_param_t()
            skp.num_points = 100
            for _ in range(skp.num_points):
                skp.contact_pose.append(lcm_utils.list2pose_stamped_lcm([0, 0, 0, 0, 0, 0, 1]))
            skp.subgoal_pose = lcm_utils.list2pose_stamped_lcm([1, 1, 1, 1, 0, 0, 0])
            skp.mask_probs = np.random.rand(skp.num_points).tolist()
            pub_msg.skill_parameter_array.append(skp)            

        self.lc.publish(self.pub_msg_name, pub_msg.encode())

if __name__ == "__main__":
    predictor = PredictorBaseLCM('test_sub', 'test_pub')
    try:    
        while True:
            # predictor.infer_single()
            predictor.infer_array()
    except KeyboardInterrupt:
        pass
    
    print('done')
