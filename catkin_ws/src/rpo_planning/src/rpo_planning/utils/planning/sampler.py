import os, os.path as osp
import time
import numpy as np
import lcm

from airobot import set_log_level, log_debug, log_info, log_warn, log_critical

import sys
import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
from rpo_lcm import (
    point_t, quaternion_t, pose_stamped_t, point_cloud_t, 
    skill_param_t, skill_param_array_t, dual_pose_stamped_t)
from rpo_planning.utils import common as util
from rpo_planning.utils import lcm_utils

class SamplerBaseLCM(object):
    def __init__(self, prefix='', timeout=60, array=True):
        # self.pub_msg_name = pub_msg_name
        # self.sub_msg_name = sub_msg_name
        self.pub_msg_name = prefix + 'env_observations'
        self.sub_msg_name = prefix + 'model_predictions'
        self.lc = lcm.LCM()
        if not array:
            self.single_subscription = self.lc.subscribe(self.sub_msg_name, self.single_handler)
        else:   
            self.array_subscription = self.lc.subscribe(self.sub_msg_name, self.array_handler)
        self.samples_count = 0
        self.model_path = None
        self.timeout = timeout

    def lcm_pub_sub_single(self, state, n_pts=100):
        self.samples_count += 1
        pointcloud_pts = state[:n_pts]

        pub_msg = lcm_utils.np2point_cloud_t(pointcloud_pts)
        pub_msg.header.seq = self.samples_count
        pub_msg.header.utime = time.time() / 1e6

        log_debug('Skill sampler sending LCM message (single) to channel: %s' % self.pub_msg_name)
        self.lc.publish(self.pub_msg_name, pub_msg.encode())

        self.received_single_data = False
        start_sub_time = time.time()
        while not self.received_single_data:
            self.lc.handle()
            if time.time() - start_sub_time > self.timeout:
                raise RuntimeError('Could not receive message in time, exiting')
        
        tc, tp, mask = self.contact_pose, self.subgoal_pose, self.subgoal_mask
        contact = []
        for i in range(n_pts):
            contact.append(util.pose_stamped2list(tc[i]))
        subgoal = util.pose_stamped2list(tp)
        return contact, subgoal, mask
        
    def lcm_pub_sub_array(self, state, n_pts=100):
        self.samples_count += 1
        pointcloud_pts = state[:n_pts]

        pub_msg = lcm_utils.np2point_cloud_t(pointcloud_pts)
        pub_msg.header.seq = self.samples_count
        pub_msg.header.utime = time.time() / 1e6

        log_debug('Skill sampler sending LCM message (array) to channel: %s' % self.pub_msg_name)
        self.lc.publish(self.pub_msg_name, pub_msg.encode())

        self.received_array_data = False
        start_sub_time = time.time()
        while not self.received_array_data:
            self.lc.handle()
            if time.time() - start_sub_time > self.timeout:
                raise RuntimeError('Could not receive message in time, exiting')
        
        contacts, subgoals, masks = [], [], []
        n_preds = self.n_preds
        for i in range(n_preds):
            tp = self.skill_parameters[i].subgoal_pose
            mask = self.skill_parameters[i].mask_probs
            for j in range(n_pts):
                tc = self.skill_parameters[i].contact_pose[j]
                contacts.append(lcm_utils.pose_stamped2list(tc.right_pose) + lcm_utils.pose_stamped2list(tc.left_pose))
            subgoals.append(lcm_utils.pose_stamped2list(tp))
            masks.append(mask)
        
        predictions = {}
        predictions['palm_predictions'] = np.asarray(contacts).reshape(n_preds, n_pts, -1)
        predictions['trans_predictions'] = np.asarray(subgoals).reshape(n_preds, -1)
        predictions['mask_predictions'] = np.asarray(masks).reshape(n_preds, n_pts, -1)
        return predictions 

    def single_handler(self, channel, data):
        msg = skill_param_t.decode(data)
        contact_pose = msg.contact_pose
        subgoal_pose = msg.subgoal_pose
        subgoal_mask = msg.mask_probs
        log_debug('Skill sampler Received LCM message from channel: %s' % channel)

        self.contact_pose = contact_pose
        self.subgoal_pose = subgoal_pose
        self.subgoal_mask = subgoal_mask
        self.received_single_data = True

    def array_handler(self, channel, data):
        msg = skill_param_array_t.decode(data)
        skill_parameters = msg.skill_parameter_array
        log_debug('Skill sampler Received LCM message from channel: %s' % channel)

        self.n_preds = msg.num_entries
        self.skill_parameters = skill_parameters
        self.received_array_data = True


if __name__ == "__main__":
    def worker(pts, q, prefix):
        sampler = SamplerBaseLCM(array=True, prefix=prefix)
        result = sampler.lcm_pub_sub_array(pts)
        print('got result!', result)
        q.put(result)

    pts = np.random.rand(100, 3)
    # sampler = SamplerBaseLCM(array=False)
    # contact, subgoal, mask = sampler.lcm_pub_sub_single(pts)
    # print(' Contact Pose: ', contact)
    # print(' Subgoal Pose: ', subgoal)
    # print(' Subgoal Mask: ', mask)

    # sampler = SamplerBaseLCM(array=True)
    # from IPython import embed
    # embed()

    from multiprocessing import Process, Pool, Queue
    q1, q2 = Queue(), Queue()

    p1 = Process(target=worker, args=(pts, q1, 'grasp_0_vae_'))
    p2 = Process(target=worker, args=(pts, q2, 'grasp_1_vae_'))
    p1.start()
    p2.start()

    r1 = q1.get()
    r2 = q2.get()

    p1.join()
    p2.join()

    
    from IPython import embed
    embed()
