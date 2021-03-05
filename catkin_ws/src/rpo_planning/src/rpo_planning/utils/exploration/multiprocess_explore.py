import time
import numpy as np
from multiprocessing import Pipe, Queue, Manager, Process

from airobot import set_log_level, log_debug, log_info, log_warn, log_critical

from rpo_planning.skills.samplers.pull import PullSamplerBasic, PullSamplerVAE
from rpo_planning.skills.samplers.push import PushSamplerBasic, PushSamplerVAE
from rpo_planning.skills.samplers.grasp import GraspSamplerBasic, GraspSamplerVAE
from rpo_planning.skills.primitive_skills import (
    GraspSkill, PullRightSkill, PullLeftSkill, PushRightSkill, PushLeftSkill
)
from rpo_planning.motion_planning.primitive_planners import (
    grasp_planning_wf, pulling_planning_wf, pushing_planning_wf
)
from rpo_planning.pointcloud_planning.rpo_learner import PointCloudTreeLearner


def worker_planner(child_conn, work_queue, result_queue, global_result_queue, global_dict, worker_flag_dict, seed, worker_id):
    while True:
        try:
            if not child_conn.poll(0.0001):
                continue
            msg = child_conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if msg == "INIT":
            prefix_n = worker_id
            skill_names = global_dict['skill_names']

            # prefixes will be used to name the LCM publisher/subscriber channels
            # the NN server will expect the same number of workers, and will name it's pub/sub
            # messages according to the worker_id -- here is where we name the sampler-side
            # interface accordingly
            pull_prefix = 'pull_%d_vae_' % prefix_n
            grasp_prefix = 'grasp_%d_vae_' % prefix_n
            push_prefix = 'push_%d_vae_' % prefix_n
            pull_sampler = PullSamplerVAE(sampler_prefix=pull_prefix)
            push_sampler = PushSamplerVAE(sampler_prefix=push_prefix)
            grasp_sampler = GraspSamplerVAE(sampler_prefix=grasp_prefix, default_target=None)

            # here we pass robot=None to each skill, since we are not performing full kinematic checking/collision free planning
            # we are only planning using the subgoals and contact poses produced by the samplers, and the robot is only used
            # for it's interface to move_group, so we don't need it here
            pull_right_skill = PullRightSkill(
                pull_sampler,
                robot=None,
                get_plan_func=pulling_planning_wf,
                ignore_mp=False,
                avoid_collisions=True
            )

            pull_left_skill = PullLeftSkill(
                pull_sampler,
                robot=None,
                get_plan_func=pulling_planning_wf,
                ignore_mp=False,
                avoid_collisions=True
            )

            push_right_skill = PushRightSkill(
                push_sampler,
                robot=None,
                get_plan_func=pushing_planning_wf,
                ignore_mp=False,
                avoid_collisions=True
            )

            push_left_skill = PushLeftSkill(
                push_sampler,
                robot=None,
                get_plan_func=pushing_planning_wf,
                ignore_mp=False,
                avoid_collisions=True
            )

            grasp_skill = GraspSkill(grasp_sampler, robot=None, get_plan_func=grasp_planning_wf)
            grasp_pp_skill = GraspSkill(grasp_sampler, robot=None, get_plan_func=grasp_planning_wf, pp=True)

            skills = {}
            for name in skill_names:
                skills[name] = None
            skills['pull_right'] = pull_right_skill
            skills['pull_left'] = pull_left_skill
            skills['grasp'] = grasp_skill
            skills['grasp_pp'] = grasp_pp_skill            
            # skills['push_right'] = push_right_skill
            # skills['push_left'] = push_left_skill
            continue
        if msg == "RESET":
            continue
        if msg == "SAMPLE":        
            worker_flag_dict[worker_id] = False
            # get the task information from the work queue
            # point_cloud, transformation, skeleton
            planner_inputs = work_queue.get()
            pointcloud_sparse = planner_inputs['pointcloud_sparse']
            pointcloud = planner_inputs['pointcloud']
            transformation_des = planner_inputs['transformation_des']
            plan_skeleton = planner_inputs['plan_skeleton']
            max_steps = planner_inputs['max_steps']
            timeout = planner_inputs['timeout']

            # setup planner
            # TODO: check if we're going to face problems using skeleton_policy = None
            # (which will not be possible if we want to predict high level skills on a per-node basis)
            planner = PointCloudTreeLearner(
                start_pcd=pointcloud_sparse,
                start_pcd_full=pointcloud,
                trans_des=transformation_des,
                plan_skeleton=plan_skeleton,
                skills=skills,
                max_steps=max_steps,
                skeleton_policy=None,
                skillset_cfg=skillset_cfg,
                pointcloud_surfaces=surfaces,
                motion_planning=False,
                timeout=timeout,
                epsilon=0.9
            )

            log_debug('Planning from RPO_Planning worker ID: %d' % worker_id)
            plan = planner.plan_with_skeleton_explore()

            result = {} 
            result['worker_id'] = worker_id
            if plan is None:
                log_info('Plan not found from worker ID: %d' % worker_id)
                result['data'] = None 
            else:
                # get transition info from the plan that was obtained
                transition_data = planner.process_plan_transitions(plan[1:])
                log_debug('Putting transition data for replay buffer into queue from worker ID %d' % worker_id)
                result['data'] = transition_data
            global_result_queue.put(result)
            worker_flag_dict[worker_id] = True 
            continue
        if msg == "END":
            break        
        time.sleep(0.001)
    log_info('Breaking Worker ID: %d' % worker_id)
    child_conn.close()


class PlanningWorkerManager:
    """
    Class to interface with a set of workers running in multiple processes
    using multiprocessing's Pipes, Queues, and Managers. In this case, workers
    are each individual instances of the RPO planner, which takes in a point cloud,
    task specification, and plan skeleton, and returns either a sequence of 
    subgoals and contact poses that reaches the goal or a flag that says the plan
    skeleton is infeasible.

    Attributes:
        global_result_queue (Queue): Queue for the workers to put their results in
        global_manager (Manager): Manager for general purpose shared memory. Used primarily
            to share a global dictionary among the workers
        global_dict (dict): Dictionary with shared global memory among the workers, for
            general-purpose data that should be accessible by all workers
        work_queues (dict): Dictionary keyed by worker id holding Queues for sending worker-specific
            data to the process
        result_queues (dict): Dictionary keyed by worker id holding Queues for receiving worker-specific
            data
        worker_flag_dict (dict): Dictionary with shared global memory among the workers,
            specifically used to flag when workers are ready for a task or have completed a task
    """
    def __init__(self, global_result_queue, global_manager, skill_names, num_workers=1):

        self.global_result_queue = global_result_queue
        self.global_manager = global_manager
        self.global_dict = self.global_manager.dict()
        self.global_dict['trial'] = 0
        self.global_dict['skill_names'] = skill_names
        self.worker_flag_dict = self.global_manager.dict()        

        self.np_seed_base = 1
        self.setup_workers(num_workers)

    def setup_workers(self, num_workers):
        """Setup function to instantiate the desired number of
        workers. Pipes and Processes set up, stored internally,
        and started.
        Args:
            num_workers (int): Desired number of worker processes
        """
        worker_ids = np.arange(num_workers, dtype=np.int64).tolist()
        seeds = np.arange(self.np_seed_base, self.np_seed_base + num_workers, dtype=np.int64).tolist()

        self._worker_ids = worker_ids
        self.seeds = seeds

        self._pipes = {}
        self._processes = {}
        self.work_queues = {}
        self.result_queues = {}
        for i, worker_id in enumerate(self._worker_ids):
            parent, child = Pipe(duplex=True)
            work_q, result_q = Queue(), Queue()
            self.work_queues[worker_id] = work_q
            self.result_queues[worker_id] = result_q
            self.worker_flag_dict[worker_id] = True
            proc = Process(
                target=worker_planner,
                args=(
                    child,
                    work_q,
                    result_q,
                    self.global_result_queue,
                    self.global_dict,
                    self.worker_flag_dict,
                    seeds[i],
                    worker_id,
                )
            )
            pipe = {}
            pipe['parent'] = parent
            pipe['child'] = child

            self._pipes[worker_id] = pipe
            self._processes[worker_id] = proc

        for i, worker_id in enumerate(self._worker_ids):
            self._processes[worker_id].start()
            self._pipes[worker_id]['parent'].send('INIT')
            log_debug('RESET WORKER ID: %d' % worker_id)
        log_debug('FINISHED WORKER SETUP')

    def put_worker_work_queue(self, worker_id, data):
        """Setter function for putting things inside the work queue
        for a specific worker

        Args:
            worker_id (int): Worker id to place data inside, for sending to worker
            data (dict): Dictionary with data to put inside of work queue. Data
                should contain information that planner needs to set up it's problem
                TODO: check to see what data fields were included in this data dictionary
        """
        self.work_queues[worker_id].put(data)
        
    def _get_worker_work_queue(self, worker_id):
        """Getter function for the work queue for a specific worker

        Args:
            worker_id (int): Worker id to place data inside, for sending to worker
        """
        return self.work_queues[worker_id].get()
    
    def _put_worker_result_queue(self, worker_id, data):
        """Setter function for putting things inside the result queue
        for a specific worker

        Args:
            worker_id (int): Worker id to place data inside, for sending to worker
            data (TODO): Data to put inside of work queue 
        """
        self.result_queues[worker_id].put(data)
        
    def get_worker_result_queue(self, worker_id):
        """Getter function for the result queue for a specific worker

        Args:
            worker_id (int): Worker id to place data inside, for sending to worker
        """
        return self.result_queues[worker_id].get()
    
    def get_global_result_queue(self):
        """Getter function for the global result queue
        """
        return self.global_result_queue.get()

    def get_global_info_dict(self):
        """Returns the globally shared dictionary of data
        generation information, including success rate and
        trial number

        Returns:
            dict: Dictionary of global information shared
                between workers
        """
        return self.global_dict

    def sample_worker(self, worker_id):
        """Function to send the "sample" command to a specific worker

        Args:
            worker_id (int): ID of which worker we should tell to start running
        """
        self._pipes[worker_id]['parent'].send('SAMPLE')

    def stop_all_workers(self):
        """Function to send an "exit" signal to all workers for clean break
        """
        for worker_id in self._worker_ids:
            self._pipes[worker_id]['parent'].send('END')

    def get_pipes(self):
        return self._pipes

    def get_processes(self):
        return self._processes

    def get_worker_ids(self):
        return self._worker_ids

    def get_worker_ready(self, worker_id):
        return self.worker_flag_dict[worker_id]
