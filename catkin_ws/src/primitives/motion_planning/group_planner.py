import moveit_commander
import moveit_msgs
import rospy
from helper import util
from moveit_commander.exception import MoveItCommanderException

# GroupPlanner interfaces planning functions for a planning group from MoveIt!
# (e.g. left_arm, right_arm or both_arms)

class GroupPlanner:
    def __init__(self, arm, robot, planner_id, scene, max_attempts, planning_time, goal_tol=0.0005,
                 eef_delta=0.01, jump_thresh=15.0):
        self.arm = arm
        self.robot = robot

        # Maximum attempts for computing a Cartesian waypoint trajectory
        self.max_attempts = max_attempts
        # Distance/step in Cartesian space for returned points in a trajectory
        self.eef_delta = eef_delta
        # Maximum jump/discontinuity (0 is any allowed)
        self.jump_thresh = jump_thresh

        # MoveGroup from MoveIt!
        self.planning_group = moveit_commander.MoveGroupCommander(self.arm)
        self.planning_group.set_goal_tolerance(goal_tol)
        self.planning_group.set_planning_time(planning_time)
        self.planning_group.set_pose_reference_frame("yumi_body")
        self.planning_group.set_planner_id(planner_id)
        self.planning_group.set_num_planning_attempts(5)
        self.planning_group.allow_replanning(True)

        # Collision scene from MoveIt!
        self.scene = scene

        # Calibration bar from YuMi table
        self.scene.add_box('calibration_bar',
                           util.list2pose_stamped([0.0, 0.495, 0.045, 0.0, 0.0, 0.0, 1.0], "yumi_body"),
                           size=(0.80, 0.05, 0.09))

        # Fake planes to limit workspace and avoid weird motions (set workspace didn't work)
        self.scene.add_plane('top',
                             util.list2pose_stamped(
                                 [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0], "yumi_body"),
                             normal=(0, 0, 1))
        self.scene.add_plane('left',
                             util.list2pose_stamped(
                                 [0.0, 0.65, 0.0, 0.0, 0.0, 0.0, 1.0], "yumi_body"),
                             normal=(0, 1, 0))
        self.scene.add_plane('right',
                             util.list2pose_stamped(
                                 [0.0, -0.65, 0.0, 0.0, 0.0, 0.0, 1.0], "yumi_body"),
                             normal=(0, 1, 0))

    # Sets start state to: (a) a defined state, if it exists, or (b) current state
    def set_start_state(self, last_trajectory=None):
        # If explicitly defined, take the last point of the last trajectory
        # Otherwise, take the current state from self.robot
        if last_trajectory is not None:
            start_state_msg = moveit_msgs.msg.RobotState()
            start_state_msg.joint_state.name = last_trajectory.joint_names
            start_state_msg.joint_state.position = last_trajectory.points[-1].positions
            self.planning_group.set_start_state(start_state_msg)
        else:
            self.planning_group.set_start_state(self.robot.get_current_state())

    # Plan for a sequence of Cartesian pose waypoints
    def plan_waypoints(self, waypoints, last_trajectory, avoid_collisions=False):
        # If first and last waypoints are the same, we assume that all waypoints are the same (the last)
        # Otherwise, we compute the entire trajectory
        if waypoints[0] == waypoints[-1]:
            return self.plan_pose_target(waypoints[-1], last_trajectory)
        else:
            # Keep trying until fraction is 1.0 or too many failed attempts
            # (fraction is the fraction of waypoints that have been achieved)
            self.set_start_state(last_trajectory)
            for i in range(self.max_attempts):
                try:
                    plan, fraction = self.planning_group.compute_cartesian_path(waypoints,
                                                                                self.eef_delta,
                                                                                self.jump_thresh,
                                                                                avoid_collisions=avoid_collisions)
                    if fraction == 1.0 and len(plan.joint_trajectory.points) > 1:
                        return plan.joint_trajectory
                except MoveItCommanderException as ex:
                    rospy.logwarn('MoveIt exception: %s. Retrying.', ex)
                    pass
            raise ValueError('Too many unsuccessful attempts.')

    # Plan for a single joint configuration target
    def plan_joint_target(self, joint_target, last_trajectory):
        for i in range(self.max_attempts):
            try:
                self.set_start_state(last_trajectory)
                self.planning_group.set_joint_value_target(joint_target)
                plan = self.planning_group.plan()
                if len(plan.joint_trajectory.points) > 1:
                    return plan.joint_trajectory
            except MoveItCommanderException as ex:
                rospy.logwarn('MoveIt exception: %s. Retrying.', ex)
                pass
        raise ValueError('Joint target seems invalid.')

    # Plan for a single Cartesian pose target
    def plan_pose_target(self, pose_target, last_trajectory):
        for i in range(self.max_attempts):
            try:
                self.set_start_state(last_trajectory)
                self.planning_group.set_joint_value_target(util.convert_pose_type(pose_target, "PoseStamped", "yumi_body"))
                plan = self.planning_group.plan()
                if len(plan.joint_trajectory.points) > 1:
                    return plan.joint_trajectory
            except MoveItCommanderException as ex:
                rospy.logwarn('MoveIt exception: %s. Retrying.', ex)
                pass
        raise ValueError('Pose target seems invalid.')
