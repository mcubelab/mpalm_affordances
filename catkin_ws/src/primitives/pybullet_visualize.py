from planning import pushing_planning, grasp_planning, levering_planning, pulling_planning
from helper import util, planning_helper, collisions

import os
from example_config import get_cfg_defaults

import airobot as ar
import time
import argparse

# def init_box():


def main(args):

	yumi = ar.create_robot('yumi',
						   robot_cfg={'render': True, 'self_collision': True})
	yumi.go_home()

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

	elif args.primitive == 'grasp':
		plan = grasp_planning(
			object=manipulated_object,
			object_pose1_world=object_pose1_world,
			object_pose2_world=object_pose2_world,
			palm_pose_l_object=palm_pose_l_object,
			palm_pose_r_object=palm_pose_r_object)

	elif args.primitive == 'pivot':
		gripper_name = args.config_package_path + \
		    'descriptions/meshes/mpalm/mpalms_all_coarse.stl'
		table_name = args.config_package_path + \
		    'descriptions/meshes/table/table_top.stl'

		manipulated_object = collisions.CollisionBody(
			args.config_package_path + 'descriptions/meshes/objects/realsense_box_experiments.stl')

		plan = levering_planning(
			object=manipulated_object,
			object_pose1_world=object_pose1_world,
			object_pose2_world=object_pose2_world,
			palm_pose_l_object=palm_pose_l_object,
			palm_pose_r_object=palm_pose_r_object,
			gripper_name=gripper_name,
			table_name=table_name)

	elif args.primitive == 'pull':
		plan = pulling_planning(
			object=manipulated_object,
			object_pose1_world=object_pose1_world,
			object_pose2_world=object_pose2_world,
			palm_pose_l_object=palm_pose_l_object,
			palm_pose_r_object=palm_pose_r_object,
			arm='r')

	else:
		raise NotImplementedError

	for plan_dict in plan:
		for i, t in enumerate(plan_dict['t']):
			tip_poses = plan_dict['palm_poses_world'][i]

			tip_to_wrist = util.list2pose_stamped(cfg.TIP_TO_WRIST_TF, '')
			world_to_world = util.unit_pose()

			wrist_left = util.convert_reference_frame(
				tip_to_wrist,
				world_to_world,
				tip_poses[0],
				"yumi_body")
			wrist_right = util.convert_reference_frame(
				tip_to_wrist,
				world_to_world,
				tip_poses[1],
				"yumi_body")

			wrist_left = util.pose_stamped2list(wrist_left)
			wrist_right = util.pose_stamped2list(wrist_right)

			r_joints = yumi.compute_ik(
				wrist_right[0:3],
				wrist_right[3:],
				arm='right',
				nullspace=True)[:7]

			l_joints = yumi.compute_ik(
				wrist_left[0:3],
				wrist_left[3:],
				arm='left',
				nullspace=True)[7:]

			t = 0.1
			start = time.time()
			while (time.time() - start < t):
				yumi.set_jpos(r_joints, arm='right', wait=False)
				time.sleep(0.005)
				yumi.set_jpos(l_joints, arm='left', wait=False)
				time.sleep(0.005)

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
