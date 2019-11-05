import rospy, numpy, random
import abb_robotnode.srv as srv
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose
import numpy as np

class Yumi:
    def __init__(self):
        self.left_arm = YumiArm('left')
        self.right_arm = YumiArm('right')

    def set_speed(self, speed_tcp, speed_ori, joints=180):
        self.left_arm.set_speed(speed_tcp, speed_ori, joints)
        self.right_arm.set_speed(speed_tcp, speed_ori, joints)

    def start_egm(self):
        self.left_arm.start_egm()
        self.right_arm.start_egm()

    def stop_egm(self):
        self.left_arm.stop_egm()
        self.right_arm.stop_egm()

    def execute_synchro_joints_trajectory(self, joints_traj, deg):
        self.left_arm.execute_joints_trajectory(joints_traj[0], synchro=True, deg=deg)
        self.right_arm.execute_joints_trajectory(joints_traj[1], synchro=True, deg=deg)

    def execute_egm_joints(self, joints, deg):
        self.left_arm.execute_egm_joints(joints[0], deg=deg)
        self.right_arm.execute_egm_joints(joints[1], deg=deg)

    def get_joints(self):
        joints_left = self.left_arm.get_joints()
        joints_right = self.right_arm.get_joints()
        return np.array([joints_left, joints_right])

class YumiArm:
    def __init__(self, arm):
        self.arm = arm
        self.name = 'robot1' if arm == 'right' else 'robot2'
        self.timeout = 0.5

        self.targetJointsPub = rospy.Publisher("/{}_TargetJoints".format(self.name), JointState, queue_size=20)
        self.setEGMMode = rospy.ServiceProxy('/{}_SetEGMMode'.format(self.name), srv.Service_SetEGMMode)

    def set_speed(self, speed_tcp, speed_ori, joints=180):
        setSpeed = rospy.ServiceProxy('/{}_SetMaxSpeed'.format(self.name), srv.Service_SetMaxSpeed)
        rospy.wait_for_service('/{}_SetMaxSpeed'.format(self.name), timeout=self.timeout)
        setSpeed(speed_tcp, speed_ori, joints)

    def start_egm(self):
        # EGM mode will be 5 (joint positions, manual)
        rospy.wait_for_service('/{}_SetEGMMode'.format(self.name), timeout=self.timeout)
        self.setEGMMode(5)

    def stop_egm(self):
        # EGM mode will be 0 (off)
        rospy.wait_for_service('/{}_SetEGMMode'.format(self.name), timeout=self.timeout)
        self.setEGMMode(0)

    def execute_joints_trajectory(self, joints_traj, synchro=False, deg=False):
        if len(joints_traj) == 1:
            joints_traj.append(joints_traj[0])
        clearBuffer = rospy.ServiceProxy('/{}_ClearJointBuffer'.format(self.name), srv.Service_ClearJointBuffer)
        addToBuffer = rospy.ServiceProxy('/{}_AddToJointBuffer'.format(self.name), srv.Service_AddToJointBuffer)
        if synchro:
            executeBuffer = rospy.ServiceProxy('/{}_ExecuteSynchroJointBuffer'.format(self.name), srv.Service_ExecuteSynchroJointBuffer)
        else:
            executeBuffer = rospy.ServiceProxy('/{}_ExecuteJointBuffer'.format(self.name), srv.Service_ExecuteJointBuffer)
        rospy.wait_for_service('/{}_ClearJointBuffer'.format(self.name), timeout=self.timeout)
        rospy.wait_for_service('/{}_AddToJointBuffer'.format(self.name), timeout=self.timeout)
        rospy.wait_for_service('/{}_ExecuteJointBuffer'.format(self.name), timeout=self.timeout)

        clearBuffer()
        for point in joints_traj:
            if deg:
                addToBuffer(point)
            else:
                addToBuffer([joint / np.pi * 180.0 for joint in point])
        executeBuffer()

    def execute_egm_joints(self, joints, deg):
        joint_msg = JointState()
        joint_msg.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
        if deg:
            joint_msg.position = joints
        else:
            joint_msg.position = [joint / np.pi * 180.0 for joint in joints]
        self.targetJointsPub.publish(joint_msg)

    def get_cartesian(self):
        '''Get the cartesian pose of the arm
        @param If true, return pose as a quaternion [3D xyz + 3D quaternion].
               Otherwise return 4x4 transformation matrix
        @param pose quaternion or matrix'''
        getCartesianRos = rospy.ServiceProxy('/{}_GetCartesian'.format(self.name), srv.Service_GetCartesian)
        rospy.wait_for_service('/{}_GetCartesian'.format(self.name), timeout=0.5)
        ret = getCartesianRos()
        pose = [ret.x, ret.y, ret.z, ret.q0, ret.qx, ret.qy, ret.qz]
        return np.array(pose)

    def get_joints(self):
        '''Send the command to return the joint poses of the arm
        @param If true, return joints in degrees. Otherwise use radians
        @return joint 7D array of the joint poses'''
        getJointRos = rospy.ServiceProxy('/{}_GetJoints'.format(self.name), srv.Service_GetJoints)
        rospy.wait_for_service('/{}_GetJoints'.format(self.name), timeout=0.5)
        js = getJointRos()
        return np.array(js.joints)

    def set_cartesian(self, pose):
        '''Set the cartesian pose of the robot's end effector.
        @param pose Desired position either given by position+quaternion or
                    tranformation matrix'''

        setCart = rospy.ServiceProxy('/{}_SetCartesian'.format(self.name), srv.Service_SetCartesian)
        rospy.wait_for_service('/{}_SetCartesian'.format(self.name), timeout=0.5)
        setCart(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6])


    def set_joints(self, joints, deg=True):
        '''Send the command to set the arm to the specified joint pose
        @param pose 7D array of arm poses'''
        if deg==False:
            joints = [joints[i] * 180 / np.pi for i in range(7)]
        if len(joints) != 7:
            raise ValueError("Arm pose must be of length 7")

        setJointRos = rospy.ServiceProxy('/{}_SetJoints'.format(self.name), srv.Service_SetJoints)
        rospy.wait_for_service('/{}_SetJoints'.format(self.name), timeout=0.5)
        setJointRos(joints)




if __name__ == "__main__":
    print ('test')