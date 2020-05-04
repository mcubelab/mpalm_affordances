import numpy as np


class PrimitiveSkill(object):
    def __init__(self, sampler):
        """Base class for primitive skill

        Args:
            sampler (function): sampling function that generates new
                potential state to add to the plan
        """
        self.sampler = sampler

    def satisfies_preconditions(self, state):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class GraspSkill(PrimitiveSkill):
    def __init__(self, sampler):
        super(GraspSkill, self).__init__(sampler)

    def sample(self, target_surface):
        # NN sampling, point cloud alignment

    def satisfies_preconditions(self, state):
        # test 1: on the table
        # test 2: in grasp region


class PullSkill(PrimitiveSkill):
    def __init__(self, sampler):
        super(PullSkill, self).__init__()


class PointCloudNode(object):
    def __init__(self):
        self.parent = None
        self.pointcloud = None
        self.transformation = None
        self.transformation_to_go = None

class PointCloudTree(self):
    def __init__(self, skeleton, skills):
        self.skeleton = skeleton
        self.skills = skills
        self.goal_threshold = None

        self.buffers = {}
        for skill in skeleton:
            self.buffers[skill] = []
        self.buffers['start'] = []
        self.buffers['final'] = []

    def plan(self, start_pcd, trans_des):
        done = False
        while not done:
            for i, skill in enumerate(self.skeleton):
                if i == 0:
                    # sample from first skill if starting at beginning
                    sample = self.skills[skill].sample(start)
                else:
                    # sample from the buffers we have
                    if not self.buffers[skill].empty():
                        state = self.buffers[skill].sample()
                        sample = self.skills[skill].sample(state)

                # if we're not at the last step, check if we can add to buffer
                if i < len(self.skeleton) - 1:
                    # check if this is a valid transition (motion planning)
                    valid = self.skills[skill].feasible(sample)

                    # check if this satisfies the constraints of the next skill
                    valid = valid and self.skills[skill+1].satisfies_preconditions(sample)
                    if valid:
                        self.buffers[skill].append(sample)
                else:
                    # sample is the proposed end state, which has the path encoded
                    # via all its parents
                    dist_to_goal = get_plan_cost(sample)
                    if dist_to_goal < self.goal_threshold:
                        done = True
                        self.buffers['final'].append(sample)
                        break
        
        # extract plan
        plan = self.extract_plan(self.buffers)
    
    def extract_plan(self, buffer):
        parent = self.buffer['final'].parent
        node = parent
        plan = []
        plan.append(node)
        while parent is not None:
            parent = node.parent
            node = parent
            plan.append(node)
        plan.append(node)
        return plan