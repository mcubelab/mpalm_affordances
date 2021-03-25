import os.path as osp
import sys
from multiprocessing import Process, Queue
from threading import Thread
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler

import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
from rpo_lcm import rpo_plan_skeleton_t


class PredictionServerInit:
    def __init__(self, lc, skill_lang, pub_name='global_skill2index', sub_name='rpo_ready_flag'):
        self.ready = False
        # self.lc = lcm.LCM()
        self.lc = lc
        self.pub_name = pub_name
        self.sub_name = sub_name
        self.skill_lang = skill_lang
        self.sub = self.lc.subscribe(sub_name, self.wait_for_ready_planner)

    def initialize(self):
        self.send_global_language(self.pub_name, self.skill_lang.skill2index)
        while True:
            self.lc.handle()
            if self.ready:
                break
        return True  # TODO: implement something that can catch an error on this comm establishment

    def send_global_language(self, pub_msg_name, skill2index):
        """
        Function to send to the planner once so that a shared skill name
        to categorical index mapping is known on both sides

        Args:
            pub_msg_name (str): LCM channel name to publish to
            skill2index (dict): Dictionary holding skill to index mappings
            lc (lc.LCM): LCM handler for calling publish
        """
        skeleton_msg = rpo_plan_skeleton_t()
        skeleton_msg.skill_names.num_strings = len(skill2index.keys())
        skeleton_msg.skill_names.string_array = list(skill2index.keys())
        skeleton_msg.num_skills = len(skill2index.keys())
        skeleton_msg.skill_indices = list(skill2index.values())

        self.lc.publish(pub_msg_name, skeleton_msg.encode())

    def wait_for_ready_planner(self, channel, data):
        """
        Dummy function to wait for planner to be ready

        Args:
            channel (str): Name of LCM channel subscription
            data ([type]): [description]
        """
        self.ready = True


class SkeletonServerParams:
    def __init__(self):
        self.skill2index = None
        self.experiment_name = None

    def set_experiment_name(self, exp):
        self.experiment_name = exp
    
    def get_experiment_name(self):
        return self.experiment_name

    def set_skill2index(self, s2i):
        self.skill2index = s2i

    def get_skill2index(self):
        return self.skill2index
        

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


def serve_wrapper(server_params, port=8000):
    # Create server
    with SimpleXMLRPCServer(('localhost', port),
                            requestHandler=RequestHandler) as server:
        server.register_introspection_functions()

        # Register an instance; all the methods of the instance are
        # published as XML-RPC methods 
        server.register_instance(server_params)

        # Run the server's main loop
        server.serve_forever()