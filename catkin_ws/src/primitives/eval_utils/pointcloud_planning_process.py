
import os
import sys
import time
import argparse

sys.path.append('/root/catkin_ws/src/primitives/')
from helper.pointcloud_planning import (
    PointCloudNode, PointCloudTree,
    GraspSamplerVAE, PullSamplerBasic,
    GraspSkill, PullSkill)


class PlanningProcessor(object):
    def __init__(self, skills, skeleton):
        self.skills = skills
        self.skeleton = skeleton

    def get_plan(self, observation_file):
        # get planning inputs
        start_pcd = observation_file['start_pcd']
        trans_des = observation_file['trans_des']

        # create planner
        planner = PointCloudTree(
            start_pcd,
            trans_des,
            self.skills,
            self.skeleton)
        plan = planner.plan()
        return plan

    def write_plan(self, plan, prediction_dir):
        pass


def main(args):
    observation_avail = len(os.listdir(args.observation_dir)) > 0

    skeleton = ['pull', 'grasp']
    pull_skill = PullSkill(pull_sampler)
    grasp_skill = GraspSkill(grasp_sampler)
    skills = {}
    skills['pull'] = pull_skill
    skills['grasp'] = grasp_skill

    processor = PlanningProcessor(skills, skeleton)

    if observation_avail:
        for fname in os.listdir(args.observation_dir):
            if fname.endswith('.npz'):
                time.sleep(0.5)
                plan = processor.get_plan(fname)
                processor.write_plan(plan, args.prediction_dir)
    time.sleep(0.01)