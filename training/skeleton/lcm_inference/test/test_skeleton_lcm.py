import os, os.path as osp
import sys
import numpy as np
import lcm
import time

import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
from rpo_lcm import point_cloud_t, pose_stamped_t, string_array_t 
from rpo_planning.utils import common as util
from rpo_planning.utils import lcm_utils


class SkeletonPredictorBaseLCM(object):
    def __init__(self, pcd_sub_name='explore_pcd_obs', task_sub_name='explore_task_obs', 
                 skeleton_pub_name='explore_skill_skeleton'):
        self.pcd_sub_name = pcd_sub_name
        self.task_sub_name = task_sub_name
        self.skeleton_pub_name = skeleton_pub_name

        self.lc = lcm.LCM()
        self.pcd_sub = self.lc.subscribe(self.pcd_sub_name, self.pcd_sub_handler)
        self.task_sub = self.lc.subscribe(self.task_sub_name, self.task_sub_handler)

    def pcd_sub_handler(self, channel, data):
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
        self.received_pcd_data = True

    def task_sub_handler(self, channel, data):
        msg = pose_stamped_t.decode(data)
        self.task_pose_list = lcm_utils.pose_stamped2list(msg)
        self.received_task_data = True

    def predict(self):
        self.received_pcd_data = False
        self.received_task_data = False
        while True: 
            self.lc.handle()
            if self.received_task_data and self.received_pcd_data:
                break
        
        points = self.points
        transformation_des = self.task_pose_list

        print( 'Got points: ', points)
        print( 'Got task: ', transformation_des)
        
        predicted_skelton = ['pull-right', 'grasp', 'pull-left', 'EOS']

        skeleton_msg = string_array_t()
        skeleton_msg.num_strings = len(predicted_skelton)
        skeleton_msg.string_array = predicted_skelton

        self.lc.publish(self.skeleton_pub_name, skeleton_msg.encode())


if __name__ == "__main__":
    predictor = SkeletonPredictorBaseLCM()
    try:    
        while True:
            predictor.predict()
    except KeyboardInterrupt:
        pass
    
    print('done')
