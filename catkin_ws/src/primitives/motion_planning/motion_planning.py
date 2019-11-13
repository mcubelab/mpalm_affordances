import numpy as np
import os
import copy
import rospy
import trajectory_msgs
import moveit_commander
import moveit_msgs
from helper import helper
from group_planner import GroupPlanner
from moveit_commander.exception import MoveItCommanderException
from moveit_msgs.srv import GetStateValidity
from ik import ik_helper


class MotionPlanner:
    def __init__(self, object_name, planner_id='RRTconnectkConfigDefault', max_attempts=25, planning_time=1.0):
        # Robot commander and planning scene (for collisions) from MoveIt!
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        # Group planners for individual arms and both arms
        self.mp_left = GroupPlanner('left_arm', self.robot, planner_id, self.scene, max_attempts, planning_time)
        self.mp_right = GroupPlanner('right_arm', self.robot, planner_id, self.scene, max_attempts, planning_time)
        self.mp_both = GroupPlanner('both_arms', self.robot, planner_id, self.scene, max_attempts, planning_time)

        # State validity check service from MoveIt!
        self.sv = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
        self.sv.wait_for_service()

        # Number of maximum attempts when planning for a sequence of Cartesian poses
        self.max_attempts = max_attempts

        # Object STL filename (to enable collision avoidance)
        self.object_stl = os.environ.get("CODE_BASE") + '/catkin_ws/src/config/descriptions/meshes/objects/' + object_name
        self.object_size = (1.1, 1.1, 1.1)

    # Interface to check state validity via MoveIt!'s service
    # (ideally used after unifying individual arm trajectories; currently not used as we assume small deviations)
    def check_plan_validity(self, joint_trajectory):
        for traj_point in joint_trajectory.points:
            gsvr = moveit_msgs.srv.GetStateValidityRequest()
            gsvr.robot_state.joint_state.name = joint_trajectory.joint_names
            gsvr.robot_state.joint_state.position = traj_point.positions
            gsvr.group_name = 'both_arms'
            result = self.sv.call(gsvr)
            if not result.valid:
                return False
        return True

    # Enables or disables object collision avoidance according to a given pose
    def set_object_collision(self, object_pose = None):
        if object_pose is None:
            self.scene.remove_world_object('object')
        else:
            self.scene.add_mesh('object', object_pose, self.object_stl, size=self.object_size)

        start = rospy.get_time()
        seconds = start
        ok = False
        while (seconds - start < 3.0) and not rospy.is_shutdown():
            # Test if we are in the expected state
            if object_pose is None:
                if not ('object' in self.scene.get_known_object_names()):
                    # rospy.loginfo('Object collision check was updated (to none) in %.4f seconds.', seconds - start)
                    ok = True
                    break
                else:
                    self.scene.remove_world_object('object')
            else:
                poses = self.scene.get_object_poses(['object'])
                if 'object' in poses:
                    # Poses from collision scene are referred to world, not yumi_body
                    poses['object'].position.z -= 0.1
                    # Check if the collision object is at the desired pose
                    if abs(poses['object'].position.x-object_pose.pose.position.x) < 0.00001 \
                            and abs(poses['object'].position.y-object_pose.pose.position.y) < 0.00001 \
                            and abs(poses['object'].position.z-object_pose.pose.position.z) < 0.00001:
                        # rospy.loginfo('Object collision check was updated (to: %.2f, %.2f, %.2f) in %.4f seconds.', object_pose.pose.position.x, object_pose.pose.position.y, object_pose.pose.position.z, seconds - start)
                        ok = True
                        break
                    else:
                        self.scene.add_mesh('object', object_pose, self.object_stl, size=self.object_size)
                else:
                    self.scene.add_mesh('object', object_pose, self.object_stl, size=self.object_size)

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.01)
            seconds = rospy.get_time()

        if not ok:
            rospy.logerr('After 3 seconds, collision check with object was not updated!')

    # Copy a pose as a waypoint for a Cartesian pose trajectory
    def waypoint_from_pose(self, pose):
        # MoveIt sometimes complains with double (C++) to np.float32 conversions
        # Values are set to float64 to avoid exceptions
        waypoint = copy.deepcopy(pose)
        waypoint.position.x = np.float64(waypoint.position.x)
        waypoint.position.y = np.float64(waypoint.position.y)
        waypoint.position.z = np.float64(waypoint.position.z)
        waypoint.orientation.x = np.float64(waypoint.orientation.x)
        waypoint.orientation.y = np.float64(waypoint.orientation.y)
        waypoint.orientation.z = np.float64(waypoint.orientation.z)
        waypoint.orientation.w = np.float64(waypoint.orientation.w)
        return waypoint

    # Create a single point joint trajectory with the given joint configuration, for a given arm
    def single_joint_trajectory(self, joints, arm='r'):
        joint_traj = trajectory_msgs.msg.JointTrajectory()
        if arm == 'l':
            joint_traj.joint_names = ['yumi_joint_%d_l' % i for i in [1, 2, 7, 3, 4, 5, 6]]
        else:
            joint_traj.joint_names = ['yumi_joint_%d_r' % i for i in [1, 2, 7, 3, 4, 5, 6]]
        traj_point = trajectory_msgs.msg.JointTrajectoryPoint()
        traj_point.positions = joints
        traj_point.time_from_start = rospy.Duration(0.0)
        joint_traj.points.append(traj_point)
        return joint_traj

    # Compute joints for the entire plan
    # For every plan, options will be planned until successful, by planning:
    # (i) the first joint-targeted home plan (considering all collisions)
    # (ii) all other pose-targeted plans will be unified into a single waypoint-targeted plan (execution) and,
    #      if successful, splitted back to the original order (w/o considering object collisions)
    def compute_final_plan(self, robot_plan):
        final_plan = []
        last_trajectory = None
        start_compute = rospy.Time.now()

        # For every plan:
        for i, plan in enumerate(robot_plan):
            rospy.loginfo("Computing joints for plan %d...", i+1)
            plan_ok = False

            # For every option within that plan:
            for k, plan_option in enumerate(plan):
                primitive_name = plan_option[0]['primitive'] if 'primitive' in plan_option[0] else ''
                rospy.loginfo("Computing joints for plan %d, option %d (%s)...", i+1, k+1, primitive_name)

                # First, compute the joint-targeted home plan
                try:
                    rospy.loginfo("Computing joints for plan %d, option %d (%s: home)...", i + 1, k + 1, primitive_name)
                    home_plan = plan_option[0]
                    home_joints, home_indexes, home_objects = self.compute_joint_item(home_plan, last_trajectory)

                    # Prepare output
                    home_planned = home_plan.copy()
                    home_planned['name'] = 'home'
                    home_planned['primitive'] = primitive_name
                    home_planned['type'] = 'SetJoints'
                    home_planned['speed'] = home_plan['speed']
                    home_planned['joint_traj'] = home_joints
                    home_planned['index_traj'] = home_indexes
                    home_planned['object_traj'] = home_objects
                    home_ok = True
                except ValueError as ex:
                    rospy.logwarn("Joints failed for plan %d, option %d (%s: home): %s.", i + 1, k + 1, primitive_name, ex)
                    home_ok = False

                # If successful and there exists an execution plan, plan for it
                if home_ok and len(plan_option) > 1:
                    try:
                        rospy.loginfo("Computing joints for plan %d, option %d (%s: execution)...", i + 1, k + 1, primitive_name)
                        self.set_object_collision(None)

                        # Prepare execution waypoints for both arms
                        waypoints_left = []
                        waypoints_right = []
                        for j, plan_item in enumerate(plan_option[1:len(plan_option)]):
                            for pose_set in plan_item['poses']:
                                waypoints_left.append(self.waypoint_from_pose(pose_set[0].pose))
                                waypoints_right.append(self.waypoint_from_pose(pose_set[1].pose))

                        # Plan for them using each arm individually
                        last_traj_left = copy.deepcopy(home_planned['joint_traj'])
                        last_traj_right = copy.deepcopy(home_planned['joint_traj'])
                        if primitive_name=='levering':
                            avoid_collision = True
                            last_traj_left.points[-1].positions = last_traj_left.points[-1].positions[0:7] + tuple(helper.yumi2robot_joints(rospy.get_param('pushing/inactive/right/yumi_convention/joints'), deg=True))
                            last_traj_right.points[-1].positions = tuple(helper.yumi2robot_joints(rospy.get_param('pushing/inactive/left/yumi_convention/joints'), deg=True)) + last_traj_right.points[-1].positions[7:14]
                        else:
                            avoid_collision = True

                        execution_left = self.mp_left.plan_waypoints(waypoints_left,
                                                                     last_trajectory=last_traj_left,
                                                                     avoid_collisions=avoid_collision)
                        execution_right = self.mp_right.plan_waypoints(waypoints_right,
                                                                       last_trajectory=last_traj_right,
                                                                       avoid_collisions=avoid_collision)

                        # Convert the single pair of individual arm trajectories (execution_left, execution_right)
                        # to a sequence of dual-arm trajectories inside the original plans (plan_option[1..])
                        rospy.loginfo("Unifying computed joints for plan %d, option %d (%s: execution)...", i+1, k+1, primitive_name)
                        plan_option[1:len(plan_option)] = self.unify_joint_trajectories(execution_left, execution_right, plan_option[1:len(plan_option)])

                        # If we arrive here, then the option is ok: stop computing options
                        last_trajectory = plan_option[-1]['joint_traj']
                        final_plan.append([home_planned] + plan_option[1:len(plan_option)])
                        plan_ok = True
                        break
                    except ValueError as ex:
                        rospy.logwarn("Joints failed for plan %d, option %d (%s): %s", i+1, k+1, primitive_name, ex)

                # If there is no execution plan, but home is ok, then this option is ok: stop computing options
                elif home_ok:
                    last_trajectory = home_planned['joint_traj']
                    final_plan.append([home_planned])
                    plan_ok = True
                    break

            # If an option is ok, go to next plan; otherwise, stop
            if plan_ok:
                rospy.loginfo("Successfully computed joints for plan group %d.", i + 1)
            else:
                rospy.logerr("All plans failed for plan group %d! Following plans not computed.", i+1)
                break

        total_time = rospy.Time.now()-start_compute
        rospy.loginfo('Successfully computed all joints in %.3f seconds.', total_time.to_sec())
        return final_plan

    # Plan for a single plan item
    def compute_joint_item(self, plan_item, last_trajectory, replan=False):
        # Determine what arms are static
        if last_trajectory is not None:
            last_state = last_trajectory.points[-1].positions
        else:
            last_state = self.robot.get_current_state().joint_state.position
        if plan_item['primitive']=='pulling':
            left_arm_static = (np.linalg.norm([last_state[i] - plan_item['joints'][-1][0][i] for i in range(7)]) < 0.001)
            right_arm_static = (np.linalg.norm([last_state[7 + i] - plan_item['joints'][-1][1][i] for i in range(7)]) < 0.001)
        else:
            left_arm_static = False
            right_arm_static = False

        # Set object collision according to plan definition
        if 'check_collisions' in plan_item and not plan_item['check_collisions']:
            self.set_object_collision(None)
        else:
            object_pose = plan_item['x_star_repl'][0] if replan and 'x_star_repl' in plan_item else plan_item['x_star'][0]
            # self.set_object_collision(object_pose)

        # Check plan type
        if 'type' in plan_item and plan_item['type'] == 'SetJoints':
            # Plan trivial trajectories for the static arms
            if left_arm_static and plan_item['primitive']=='pulling':
                # Here, left arm does not move, so we plan only for the right arm
                item_plan_left = self.single_joint_trajectory(plan_item['joints'][-1][0], arm='l')
                item_plan_right = self.mp_right.plan_joint_target(plan_item['joints'][-1][1], last_trajectory)
                return self.unify_joint_trajectory(item_plan_left, item_plan_right, plan_item, replan)
            elif right_arm_static and plan_item['primitive']=='pulling':
                # Here, right arm does not move, so we plan only for the left arm
                item_plan_left = self.mp_left.plan_joint_target(plan_item['joints'][-1][0], last_trajectory)
                item_plan_right = self.single_joint_trajectory(plan_item['joints'][-1][1], arm='r')
                return self.unify_joint_trajectory(item_plan_left, item_plan_right, plan_item, replan)
            else:
                # Here, both arm move. If type is SetJoints, object remains at the same place
                joint_target = plan_item['joints'][-1][0] + plan_item['joints'][-1][1]
                joint_traj = self.mp_both.plan_joint_target(joint_target, last_trajectory)
                index_traj = [[0, 0, 0.0] for point in joint_traj.points]
                object_traj = [plan_item['x_star'][0] for point in joint_traj.points]
                return joint_traj, index_traj, object_traj

        elif 'type' in plan_item and plan_item['type'] == 'SetCart':
            # Get poses depending if it is a replanned plan or not
            pose_target_left = plan_item['poses_repl'][-1][0] if replan else plan_item['poses'][-1][0]
            pose_target_right = plan_item['poses_repl'][-1][1] if replan else plan_item['poses'][-1][1]

            # Plan for each arm and unify
            item_plan_left = self.mp_left.plan_pose_target(pose_target_left, last_trajectory)
            item_plan_right = self.mp_right.plan_pose_target(pose_target_right, last_trajectory)
            return self.unify_joint_trajectory(item_plan_left, item_plan_right, plan_item, replan)

        elif 'type' in plan_item and plan_item['type'] == 'SetTraj':
            # Prepare waypoints
            waypoints_left = []
            waypoints_right = []
            pose_list = plan_item['poses_repl'] if replan else plan_item['poses']
            for pose_set in pose_list:
                waypoints_left.append(self.waypoint_from_pose(pose_set[0].pose))
                waypoints_right.append(self.waypoint_from_pose(pose_set[1].pose))

            # Compute joint plan
            item_plan_left = self.mp_left.plan_waypoints(waypoints_left, last_trajectory)
            item_plan_right = self.mp_right.plan_waypoints(waypoints_right, last_trajectory)
            return self.unify_joint_trajectory(item_plan_left, item_plan_right, plan_item, replan)

        elif 'type' in plan_item and plan_item['type'] == 'SetJointsTraj':
            # Prepare waypoints
            waypoints_right = []
            pose_list = plan_item['poses_repl'] if replan else plan_item['poses']
            for pose_set in pose_list:
                waypoints_right.append(self.waypoint_from_pose(pose_set[1].pose))

            item_plan_left = self.mp_left.plan_joint_target(plan_item['joints'][-1][0], last_trajectory)
            item_plan_right = self.mp_right.plan_waypoints(waypoints_right, last_trajectory)
            return self.unify_joint_trajectory(item_plan_left, item_plan_right, plan_item, replan)

        elif 'type' in plan_item and plan_item['type'] == 'SetTrajJoints':
            # Prepare waypoints
            waypoints_left = []
            pose_list = plan_item['poses_repl'] if replan else plan_item['poses']
            for pose_set in pose_list:
                waypoints_left.append(self.waypoint_from_pose(pose_set[0].pose))

            item_plan_left = self.mp_right.plan_waypoints(waypoints_left, last_trajectory)
            item_plan_right = self.mp_right.plan_joint_target(plan_item['joints'][-1][1], last_trajectory)
            return self.unify_joint_trajectory(item_plan_left, item_plan_right, plan_item, replan)
        else:
            raise ValueError('No plan type found.')


    # Given a reference pose and a list of poses (optionally considering just a part of it), returns the index of
    # the closest pose to the reference (wrt to the entire list)
    def compute_closest_pose_index(self, ref_pose, pose_list, start = 0):
        difs = [np.linalg.norm([ref_pose.pose.position.x - cur_pose.pose.position.x,
                                ref_pose.pose.position.y - cur_pose.pose.position.y,
                                ref_pose.pose.position.z - cur_pose.pose.position.z]) for cur_pose in pose_list[start:len(pose_list)]]
        return start + np.argmin(difs)


    # Given a pair of individual arm trajectories (traj_left, traj_right), compute a sequence of dual-arm trajectories
    # corresponding to the given plan items, and save them into those items as plan_items[i]['joint_traj']
    def unify_joint_trajectories(self, traj_left, traj_right, plan_items):
        # start_{left,right} and end_{left,right} are the start and end indexes for each arm and the current item
        start_left = 0
        start_right = 0

        # Compute FK for all the joints
        fks_left = [ik_helper.compute_fk(list(joints.positions), arm='l') for joints in traj_left.points]
        fks_right = [ik_helper.compute_fk(list(joints.positions), arm='r') for joints in traj_right.points]

        # For every item, we will find the start and end indexes for each arm
        # Start indexes are given by the end indexes of the previous item
        for item_order, plan_item in enumerate(plan_items):
            if item_order == len(plan_items)-1:
                # Since it is the last item, take all the remaining joints
                end_left = len(traj_left.points)-1
                end_right = len(traj_right.points)-1
            else:
                # End indexes are found computing FK and finding the first most similar pose to the last item pose
                end_left = self.compute_closest_pose_index(plan_item['poses'][-1][0], fks_left, start=start_left)
                end_right = self.compute_closest_pose_index(plan_item['poses'][-1][1], fks_right, start=start_right)

            # Prepare individual subtrajectories and unify them
            # Even if all joint names are in both arm trajectories, only the active joints are considered
            item_traj_left = trajectory_msgs.msg.JointTrajectory()
            item_traj_left.joint_names = ['yumi_joint_%d_l' % i for i in [1, 2, 7, 3, 4, 5, 6]] + ['yumi_joint_%d_r' % i for i in [1, 2, 7, 3, 4, 5, 6]]
            item_traj_left.points = traj_left.points[start_left:end_left+1]
            item_traj_right = trajectory_msgs.msg.JointTrajectory()
            item_traj_right.joint_names = ['yumi_joint_%d_l' % i for i in [1, 2, 7, 3, 4, 5, 6]] + ['yumi_joint_%d_r' % i for i in [1, 2, 7, 3, 4, 5, 6]]
            item_traj_right.points = traj_right.points[start_right:end_right+1]

            plan_item['joint_traj'], plan_item['index_traj'], plan_item['object_traj'] = self.unify_joint_trajectory(item_traj_left, item_traj_right, plan_item)

            start_left = end_left
            start_right = end_right
        return plan_items

    def unify_joint_trajectory(self, traj_left, traj_right, plan_item, replan=False):
        # Unification depending if object position can be taken as a reference
        object_list = plan_item['x_star_repl'] if replan and 'x_star_repl' in plan_item else plan_item['x_star']
        if object_list[0] == object_list[-1]:
            return self.unify_joint_trajectory_scaled(traj_left, traj_right, plan_item, replan)
        else:
            return self.unify_joint_trajectory_object(traj_left, traj_right, plan_item, replan)

    def unify_joint_trajectory_scaled(self, traj_left, traj_right, plan_item, replan):
        unified_traj = trajectory_msgs.msg.JointTrajectory()
        index_traj = []
        object_traj = []

        # Prepare header
        unified_traj.joint_names = ['yumi_joint_%d_l' % i for i in [1, 2, 7, 3, 4, 5, 6]] + ['yumi_joint_%d_r' % i for i
                                                                                             in [1, 2, 7, 3, 4, 5, 6]]

        # Prepare timestamps
        start_left = traj_left.points[0].time_from_start
        start_right = traj_right.points[0].time_from_start
        duration_left = traj_left.points[-1].time_from_start-start_left
        duration_right = traj_right.points[-1].time_from_start-start_right
        max_duration = np.max([duration_left, duration_right])
        longest_seq = 'left' if duration_left >= duration_right else 'right'

        pose_list = plan_item['poses_repl'] if replan and 'poses_repl' in plan_item else plan_item['poses']
        object_list = plan_item['x_star_repl'] if replan and 'x_star_repl' in plan_item else plan_item['x_star']

        if duration_left.to_sec() > 0:
            times_l = np.array(
                [(point.time_from_start.to_sec()-start_left.to_sec()) / duration_left.to_sec() * max_duration.to_sec() for point in
                 traj_left.points])
            joints_l = np.array([point.positions for point in traj_left.points])
        else:
            times_l = np.array([0, max_duration.to_sec()])
            joints_l = np.array([traj_left.points[0].positions, traj_left.points[0].positions])
        if duration_right.to_sec() > 0:
            times_r = np.array(
                [(point.time_from_start.to_sec()-start_right.to_sec()) / duration_right.to_sec() * max_duration.to_sec() for point in
                 traj_right.points])
            joints_r = np.array([point.positions for point in traj_right.points])
            joints_r = np.array([point.positions for point in traj_right.points])
        else:
            times_r = np.array([0, max_duration.to_sec()])
            joints_r = np.array([traj_right.points[0].positions, traj_right.points[0].positions])

        # Prepare joint trajectory points
        times = sorted(set(list(times_l) + list(times_r)))
        position_left = [0., 0., 0., 0., 0., 0., 0.]
        position_right = [0., 0., 0., 0., 0., 0., 0.]
        xstar_index = 0
        for current_item_time in times:
            # Left
            times_dif_l = times_l - current_item_time
            i = np.max(np.where(times_dif_l <= 0.000001))
            ii = np.min(np.where(times_dif_l >= -0.000001))
            for j in range(7):
                position_left[j] = np.interp(current_item_time, [times_l[i], times_l[ii]],
                                             [joints_l[i][j], joints_l[ii][j]])
            if longest_seq == 'left':
                pose_left = ik_helper.compute_fk(position_left, arm="l")
                while xstar_index < len(object_list)-1 and np.linalg.norm([pose_list[xstar_index][0].pose.position.x-pose_left.pose.position.x, pose_list[xstar_index][0].pose.position.y-pose_left.pose.position.y, pose_list[xstar_index][0].pose.position.z-pose_left.pose.position.z]) > 0.0006:
                    xstar_index += 1
                index_traj.append([xstar_index, xstar_index, 0.0])
                object_traj.append(object_list[xstar_index])
            # Right
            times_dif_r = times_r - current_item_time
            i = np.max(np.where(times_dif_r <= 0.000001))
            ii = np.min(np.where(times_dif_r >= -0.000001))
            for j in range(7):
                position_right[j] = np.interp(current_item_time, [times_r[i], times_r[ii]], [joints_r[i][j], joints_r[ii][j]])
            if longest_seq == 'right':
                pose_right = ik_helper.compute_fk(position_right, arm="r")
                while xstar_index < len(object_list)-1 and np.linalg.norm([pose_list[xstar_index][1].pose.position.x-pose_right.pose.position.x, pose_list[xstar_index][1].pose.position.y-pose_right.pose.position.y, pose_list[xstar_index][1].pose.position.z-pose_right.pose.position.z]) > 0.0006:
                    xstar_index += 1
                index_traj.append([xstar_index, xstar_index, 0.0])
                object_traj.append(object_list[xstar_index])
            # Append to list
            traj_point = trajectory_msgs.msg.JointTrajectoryPoint()
            traj_point.positions = position_left + position_right
            traj_point.time_from_start = rospy.Duration(current_item_time)
            unified_traj.points.append(traj_point)
        return unified_traj, index_traj, object_traj

    def unify_joint_trajectory_object(self, traj_left, traj_right, plan_item, replan):
        unified_traj = trajectory_msgs.msg.JointTrajectory()
        index_traj = []
        object_traj = []

        # Prepare header
        unified_traj.joint_names = ['yumi_joint_%d_l' % i for i in [1, 2, 7, 3, 4, 5, 6]] + ['yumi_joint_%d_r' % i for i
                                                                                             in [1, 2, 7, 3, 4, 5, 6]]

        # Prepare timestamps
        start_left = traj_left.points[0].time_from_start
        start_right = traj_right.points[0].time_from_start
        duration_left = traj_left.points[-1].time_from_start-start_left
        duration_right = traj_right.points[-1].time_from_start-start_right
        max_duration = np.max([duration_left, duration_right])
        long_arm = 'l' if duration_left >= duration_right else 'r'
        short_arm = 'r' if duration_left >= duration_right else 'l'
        start_long = start_left if long_arm == 'l' else start_right

        long_traj = traj_left if duration_left >= duration_right else traj_right
        short_traj = traj_right if duration_left >= duration_right else traj_left
        long_arr = 0 if duration_left >= duration_right else 1
        short_arr = 1 if duration_left >= duration_right else 0

        index_input = 0
        index_short = 0

        pose_list = plan_item['poses_repl'] if replan and 'poses_repl' in plan_item else plan_item['poses']
        object_list = plan_item['x_star_repl'] if replan and 'x_star_repl' in plan_item else plan_item['x_star']
        for point in long_traj.points:
            traj_point = trajectory_msgs.msg.JointTrajectoryPoint()
            joints_long = list(point.positions)
            pose_long = ik_helper.compute_fk(joints_long, arm=long_arm)

            # Localize the input index corresponding to these joints
            # This will be used for object pose + coordinate joints of the other arm
            fk_to_inputs = np.array([np.linalg.norm([input_poses[long_arr].pose.position.x-pose_long.pose.position.x, input_poses[long_arr].pose.position.y-pose_long.pose.position.y, input_poses[long_arr].pose.position.z-pose_long.pose.position.z]) for input_poses in pose_list[index_input:len(pose_list)]])
            position_dist = 0.02
            candidates = [index_input]
            new_candidates = [idx + index_input for idx in np.where(fk_to_inputs < position_dist)[0].tolist()]
            while len(new_candidates) > 0 and position_dist > 0.0005:
                candidates = new_candidates
                position_dist /= 2
                new_candidates = [idx + index_input for idx in np.where(fk_to_inputs < position_dist)[0].tolist()]
            index_input = np.max(candidates)

            fks_to_input = np.array([np.linalg.norm([pose_list[index_input][short_arr].pose.position.x-ik_helper.compute_fk(joints_short.positions, arm=short_arm).pose.position.x, pose_list[index_input][short_arr].pose.position.y-ik_helper.compute_fk(joints_short.positions, arm=short_arm).pose.position.y, pose_list[index_input][short_arr].pose.position.z-ik_helper.compute_fk(joints_short.positions, arm=short_arm).pose.position.z]) for joints_short in short_traj.points[index_short:len(short_traj.points)]])
            position_dist = 0.02
            candidates = [index_short]
            new_candidates = [idx + index_short for idx in np.where(fks_to_input < position_dist)[0].tolist()]
            while len(new_candidates) > 0 and position_dist > 0.0005:
                candidates = new_candidates
                position_dist /= 2
                new_candidates = [idx + index_short for idx in np.where(fks_to_input < position_dist)[0].tolist()]
            index_short = np.max(candidates)
            joints_short = [np.average([short_traj.points[k].positions[i] for k in candidates]) for i in range(7)]

            traj_point.positions = joints_long + joints_short if long_arm == 'l' else joints_short + joints_long
            traj_point.time_from_start = point.time_from_start-start_long
            unified_traj.points.append(traj_point)
            index_traj.append([index_input, index_input, 0.0])
            object_traj.append(object_list[index_input])
        return unified_traj, index_traj, object_traj


    # # Deprecated version of compute_final_plan (planning individually each item)
    # # This sometimes make following plan items unfeasible (all poses should be planned together)
    # def compute_final_plan_by_item(self, robot_plan):
    #     final_plan = []
    #     last_trajectory = None
    #
    #     for i, plan in enumerate(robot_plan):
    #         rospy.loginfo("Computing joints for plan group %d...", i+1)
    #
    #         for k, plan_option in enumerate(plan):
    #             rospy.loginfo("Computing joints for plan group %d, option %d...", i+1, k+1)
    #             option_attempt = 0
    #
    #             while not rospy.is_shutdown():
    #                 option_ok = True
    #
    #                 for j, plan_item in enumerate(plan_option):
    #                     rospy.loginfo("Computing joints for plan group %d, option %d, item %d (%s: %s)...", i + 1, k + 1, j + 1, plan_item['primitive'] if 'primitive' in plan_item else '', plan_item['name'] if 'name' in plan_item else '')
    #                     try:
    #                         plan_item['joint_traj'], plan_item['index_traj'], plan_item['object_traj'] = self.compute_joint_item(plan_item, last_trajectory)
    #                         # rospy.loginfo('Joint length: %d, object length: %d', len(plan_item['joint_traj'].points), len(plan_item['object_traj']))
    #                         last_trajectory = plan_item['joint_traj']
    #                     except ValueError as ex:
    #                         option_ok = False
    #                         last_trajectory = final_plan[-1][-1]['joint_traj'] if i > 0 else None
    #                         option_attempt += 1
    #                         if option_attempt == 25:
    #                             option_ok = False
    #                             rospy.logwarn("Joints failed for plan group %d, option %d, item %d. %s", i + 1, k + 1, j + 1, ex)
    #                         else:
    #                             rospy.logwarn("Joints failed for this attempt. Retrying: %s", ex)
    #                         break
    #
    #                 # After all items computed
    #                 if option_ok:
    #                     final_plan.append(plan_option)
    #                     rospy.loginfo("Successfully computed joints for plan group %d.", i+1)
    #                     break
    #
    #         # After all options computed
    #         if not option_ok:
    #             rospy.logerr("All plans failed for plan group %d! Following plans not computed.", i+1)
    #             break
    #
    #     return final_plan
