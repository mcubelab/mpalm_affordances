import os.path as osp
import sys
import lcm
import xmlrpc.client

from airobot import set_log_level, log_debug, log_info, log_warn, log_critical

import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
from rpo_lcm import string_array_t, rpo_plan_skeleton_t 


class PlanningClientRPC:
    def __init__(self, port=8000):
        self.addr = 'http://localhost:%d' % port 
        log_debug('Connecting to server at address: %s' % self.addr)
        self.s = xmlrpc.client.ServerProxy(self.addr)
        list_methods_msg = ', '.join([str(i) for i in self.s.system.listMethods()])
        log_debug('Available server methods: %s' % list_methods_msg)

    def get_skill2index(self):
        return self.s.get_skill2index()
    
    def get_experiment_name(self):
        return self.s.get_experiment_name()


class PlanningClientInit:
    def __init__(self, sub_name='global_skill2index', pub_name='rpo_ready_flag'):
        self.ready = False
        self.lc = lcm.LCM()
        self.pub_name = pub_name
        self.sub_name = sub_name
        self.sub = self.lc.subscribe(sub_name, self.receive_skill2index)

    def setup_global_language(self):
        while True:
            self.lc.handle()
            if self.ready:
                break
        self.send_ready_command()
        skill2index = dict(zip(self.skills, self.skill_indices))
        return skill2index

    def send_ready_command(self):
        dummy_message = string_array_t()
        dummy_message.num_strings = 1
        dummy_message.string_array = ['ready']
        self.lc.publish(self.pub_name, dummy_message.encode())

    def receive_skill2index(self, channel, data):
        """
        Dummy function to wait for planner to be ready

        Args:
            channel (str): Name of LCM channel subscription
            data ([type]): [description]
        """
        msg = rpo_plan_skeleton_t.decode(data)
        self.skills = msg.skill_names.string_array
        self.skill_indices = msg.skill_indices
        self.ready = True