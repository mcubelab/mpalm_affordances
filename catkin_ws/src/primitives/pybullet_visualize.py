from planning import pushing_planning
from helper import util, planning_helper, collisions

import os
from example_config import get_cfg_defaults

import airobot as ar
import time
import argparse

# def init_box():


def main(args):

	yumi = ar.create_robot('yumi', robot_cfg={'render': True, 'self_collision': True})
	cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
	cfg = get_cfg_defaults()
	cfg.merge_from_file(cfg_file)
	cfg.freeze()
	print(cfg)

	manipulated_object = None
	object_pose1_world = util.list2pose_stamped(cfg.OBJECT_INIT)
	object_pose2_world = util.list2pose_stamped(cfg.OBJECT_FINAL)
	palm_pose_l_object = util.list2pose_stamped(cfg.PALM_LEFT)
	palm_pose_r_object = util.list2pose_stamped(cfg.PALM_RIGHT)

	if args.primitive == 'push':
		plan = pushing_planning(
			object=manipulated_object,
			object_pose1_world=object_pose1_world,
			object_pose2_world=object_pose2_world,
			palm_pose_l_object=palm_pose_l_object,
			palm_pose_r_object=palm_pose_r_object)
	else:
		raise NotImplementedError

	for plan_dict in plan:
		for i, t in enumerate(plan_dict['t']):
			# print(plan_dict['palm_poses_world'][i])
			poses = plan_dict['palm_poses_world'][i]
			right_palm = util.pose_stamped2list(poses[0])
			left_palm = util.pose_stamped2list(poses[1])

			print(right_palm)
			r_joints = yumi.compute_ik(
				right_palm[0:3],
				right_palm[3:],
				arm='right')

			l_joints = yumi.compute_ik(
				left_palm[0:3],
				left_palm[3:],
				arm='left'
			)

			joints = r_joints + l_joints
			yumi.set_jpos(joints)
			
			# yumi.set_ee_pose(right_palm[0:3], right_palm[3:], arm='right')
			# yumi.set_ee_pose(left_palm[0:3], left_palm[3:], arm='left')
			time.sleep(0.1)

	from IPython import embed
	embed()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_package_path',
						type=str,
					 	default='/home/anthony/repos/research/mpalm_affordances/catkin_ws/src/config/')
	parser.add_argument('--example_config_path', type=str, default='config')
	parser.add_argument('--primitive', type=str, default='push', help='which primitive to plan')
	parser.add_argument('--simulate', type=bool, default=True)
	args = parser.parse_args()
	main(args)
