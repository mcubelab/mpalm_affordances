import numpy as np
import tf
# import tf.transformations
#~ from geometry_msgs.msg import PoseStamped
import math
import random
import pybullet as p



class Position:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.


class Orientation:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.w = 0.


class Pose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation


class Stamp:
    def __init__(self):
        self.secs = 0.0
        self.nsecs = 0.0


class Header:
    def __init__(self):
        self.frame_id = "world"
        self.stamp = Stamp()


class PoseStamped():
    def __init__(self):
        position = Position()
        orientation = Orientation()
        pose = Pose(position, orientation)
        header = Header()
        self.pose = pose
        self.header = header


def get_2d_pose(pose3d):
    #1. extract rotation about z-axis
    T = matrix_from_pose(pose3d)
    euler_angles_list = tf.transformations.euler_from_matrix(T, 'rxyz')
    pose2d = np.array([pose3d.pose.position.x,
                       pose3d.pose.position.y,
                       euler_angles_list[2],
                       ])
    return pose2d


def C3_2d(theta):
    C = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]]
                 )

    return C


def C3(theta):
    C = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0],
                  [0,0,1]]
                 )
    return C


def unwrap(angles, min_val=-np.pi, max_val=np.pi):
    if type(angles) is not 'ndarray':
        angles = np.array(angles)
    angles_unwrapped = []
    for counter in range(angles.shape[0]):
        angle = angles[counter]
        if angle < min_val:
            angle +=  2 * np.pi
        if angle > max_val:
            angle -=  2 * np.pi
        angles_unwrapped.append(angle)
    return np.array(angles_unwrapped)


def angle_from_3d_vectors(u, v):
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    u_dot_v = np.dot(u, v)
    return np.arccos(u_dot_v) / (u_norm * v_norm)


def pose_from_matrix(matrix, frame_id="world"):
    trans = tf.transformations.translation_from_matrix(matrix)
    quat = tf.transformations.quaternion_from_matrix(matrix)
    pose = list(trans) + list(quat)
    pose = list2pose_stamped(pose, frame_id=frame_id)
    return pose


def list2pose_stamped(pose, frame_id="world"):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = pose[0]
    msg.pose.position.y = pose[1]
    msg.pose.position.z = pose[2]
    msg.pose.orientation.x = pose[3]
    msg.pose.orientation.y = pose[4]
    msg.pose.orientation.z = pose[5]
    msg.pose.orientation.w = pose[6]
    return msg


def unit_pose():
    return list2pose_stamped([0,0,0,0,0,0,1])


def convert_reference_frame(pose_source, pose_frame_target, pose_frame_source, frame_id = "yumi_body"):
    T_pose_source = matrix_from_pose(pose_source)
    pose_transform_target2source = get_transform(pose_frame_source, pose_frame_target)
    T_pose_transform_target2source = matrix_from_pose(pose_transform_target2source)
    T_pose_target = np.matmul(T_pose_transform_target2source, T_pose_source)
    pose_target = pose_from_matrix(T_pose_target, frame_id=frame_id)
    return pose_target


def convert_reference_frame_list(pose_source_list, pose_frame_target, pose_frame_source, frame_id = "yumi_body"):
    pose_target_list = []
    for pose_source in pose_source_list:
        pose_target_list.append(convert_reference_frame(pose_source,
                                                        pose_frame_target,
                                                        pose_frame_source,
                                                        frame_id))
    return pose_target_list


def pose_stamped2list(msg):
    return [float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
            ]


def pose_stamped2np(msg):
    return np.asarray(pose_stamped2list(msg))


def get_transform(pose_frame_target, pose_frame_source):
    """
    Find transform that transforms pose source to pose target
    :param pose_frame_target:
    :param pose_frame_source:
    :return:
    """
    #both poses must be expressed in same reference frame
    T_target_world = matrix_from_pose(pose_frame_target)
    T_source_world = matrix_from_pose(pose_frame_source)
    T_relative_world = np.matmul(T_target_world, np.linalg.inv(T_source_world))
    pose_relative_world = pose_from_matrix(T_relative_world, frame_id=pose_frame_source.header.frame_id)
    return pose_relative_world


def matrix_from_pose(pose):
    pose_list = pose_stamped2list(pose)
    trans = pose_list[0:3]
    quat = pose_list[3:7]
    T = tf.transformations.quaternion_matrix(quat)
    T[0:3,3] = trans
    return T


def euler_from_pose(pose):
    T_transform = matrix_from_pose(pose)
    euler = tf.transformations.euler_from_matrix(T_transform, 'rxyz')
    return euler


def rotate_quat_y(pose):
    '''set orientation of right gripper as a mirror reflection of left gripper about y-axis'''
    quat = pose.pose.orientation
    R = tf.transformations.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
    z_vec = R[0:3, 2]  # np.cross(x_vec, y_vec)
    y_vec = -R[0:3, 1]  # normal
    x_vec = np.cross(y_vec, z_vec)  # np.array([0,0,-1])
    x_vec = x_vec / np.linalg.norm(x_vec)
    y_vec = y_vec / np.linalg.norm(y_vec)
    z_vec = z_vec / np.linalg.norm(z_vec)
    # Normalized object frame
    hand_orient_norm = np.vstack((x_vec, y_vec, z_vec))
    hand_orient_norm = hand_orient_norm.transpose()
    quat = mat2quat(hand_orient_norm)
    return quat


def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def mat2quat(orient_mat_3x3):
    orient_mat_4x4 = [[orient_mat_3x3[0][0],orient_mat_3x3[0][1],orient_mat_3x3[0][2],0],
                       [orient_mat_3x3[1][0],orient_mat_3x3[1][1],orient_mat_3x3[1][2],0],
                       [orient_mat_3x3[2][0],orient_mat_3x3[2][1],orient_mat_3x3[2][2],0],
                       [0,0,0,1]]

    orient_mat_4x4 = np.array(orient_mat_4x4)
    quat = quaternion_from_matrix(orient_mat_4x4)
    return quat


def interpolate_pose(pose_initial, pose_final, N, frac=1):
    frame_id = pose_initial.header.frame_id
    pose_initial_list = pose_stamped2list(pose_initial)
    pose_final_list = pose_stamped2list(pose_final)
    trans_initial = pose_initial_list[0:3]
    quat_initial = pose_initial_list[3:7]
     # onvert to pyquaterion convertion (w,x,y,z)
    trans_final = pose_final_list[0:3]
    quat_final = pose_final_list[3:7]

    trans_interp_total = [np.linspace(trans_initial[0], trans_final[0], num=N),
                          np.linspace(trans_initial[1], trans_final[1], num=N),
                          np.linspace(trans_initial[2], trans_final[2], num=N)]
    pose_interp = []
    for counter in range(int(frac * N)):
        quat_interp = tf.transformations.quaternion_slerp(quat_initial,
                                                          quat_final,
                                                          float(counter) / (N-1))
        pose_tmp = [trans_interp_total[0][counter],
                            trans_interp_total[1][counter],
                            trans_interp_total[2][counter],
                            quat_interp[0], #return in ROS ordering w,x,y,z
                            quat_interp[1],
                            quat_interp[2],
                            quat_interp[3],
                            ]
        pose_interp.append(list2pose_stamped(pose_tmp, frame_id=frame_id))
    return pose_interp


def offset_local_pose(pose_world, offset):
    #1. convert to gripper reference frame
    pose_gripper = convert_reference_frame(pose_world,
                                             pose_world,
                                             unit_pose(),
                                             frame_id = "local")

    #3. add offset to grasp poses in gripper frames
    pose_gripper.pose.position.x += offset[0]
    pose_gripper.pose.position.y += offset[1]
    pose_gripper.pose.position.z += offset[2]
    #4. convert back to world frame
    pose_new_world = convert_reference_frame(pose_gripper,
                                             unit_pose(),
                                             pose_world,
                                             frame_id = "world")
    return pose_new_world


def transform_pose(pose_source, pose_transform):
    T_pose_source = matrix_from_pose(pose_source)
    T_transform_source = matrix_from_pose(pose_transform)
    T_pose_final_source = np.matmul(T_transform_source, T_pose_source)
    pose_final_source = pose_from_matrix(T_pose_final_source, frame_id=pose_source.header.frame_id)
    return pose_final_source


def transform_body(pose_source_world, pose_transform_target_body):
    #convert source to target frame
    pose_source_body = convert_reference_frame(pose_source_world,
                                                 pose_source_world,
                                                 unit_pose(),
                                                 frame_id="body_frame")
    #perform transformation in body frame
    pose_source_rotated_body = transform_pose(pose_source_body,
                                              pose_transform_target_body)
    # rotate back
    pose_source_rotated_world = convert_reference_frame(pose_source_rotated_body,
                                                         unit_pose(),
                                                         pose_source_world,
                                                         frame_id="yumi_body")
    return pose_source_rotated_world


def rotate_local_pose(pose_world, offset):
    angle_x = offset[0]
    angle_y = offset[1]
    angle_z = offset[2]
    pose_transform_tmp = pose_from_matrix(tf.transformations.euler_matrix(angle_x, angle_y, angle_z, 'sxyz'),
                                                frame_id="tmp")

    pose_rotated_world = transform_body(pose_world, pose_transform_tmp)
    return pose_rotated_world


def rotate_local_pose_list(pose_world_list, offset_list):
    pose_rotated_world_list = []
    for i, pose_world in enumerate(pose_world_list):
            pose_rotated_world_list.append(rotate_local_pose(pose_world, offset_list[i]))
    return pose_rotated_world_list


def offset_local_pose(pose_world, offset):
    #1. convert to gripper reference frame
    pose_gripper = convert_reference_frame(pose_world,
                                                         pose_world,
                                                         unit_pose(),
                                                         frame_id = "gripper_frame")

    #3. add offset to grasp poses in gripper frames
    pose_gripper.pose.position.x += offset[0]
    pose_gripper.pose.position.y += offset[1]
    pose_gripper.pose.position.z += offset[2]
    #4. convert back to world frame
    pose_new_world = convert_reference_frame(pose_gripper,
                                                         unit_pose(),
                                                         pose_world,
                                                         frame_id = "yumi_body")
    return pose_new_world


def vec_from_pose(pose):
    #get unit vectors of rotation from pose
    quat = pose.pose.orientation
    R = tf.transformations.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
    x_vec = R[0:3, 0]
    y_vec = R[0:3, 1]
    z_vec = R[0:3, 2]
    return x_vec, y_vec, z_vec


def convert_pose_type(pose, type_out="list", frame_out="yumi_body"):
    type_in = type(pose)
    assert (type_in in [Pose, list, np.ndarray, PoseStamped, dict])
    assert (type_out in ["Pose", "list", "ndarray", "PoseStamped", "dict"])
    #convert all poses to list
    if type_in == Pose:
        pose = pose2list(pose)
    elif type_in == PoseStamped:
        pose = pose_stamped2list(pose)
    elif type_in == dict:
        pose = dict2pose_stamped(pose)
    #convert to proper output type
    if type_out == "Pose":
        pose_out = list2pose(pose)
    elif type_out == "PoseStamped":
        pose_out = list2pose_stamped(pose, frame_id=frame_out)
    elif type_out == "dict":
        pose_out = list2dict(pose, frame_id=frame_out)
    else:
        pose_out = pose
    return pose_out


def list_to_pose(pose_list):
    msg = Pose()
    msg.position.x = pose_list[0]
    msg.position.y = pose_list[1]
    msg.position.z = pose_list[2]
    msg.orientation.x = pose_list[3]
    msg.orientation.y = pose_list[4]
    msg.orientation.z = pose_list[5]
    msg.orientation.w = pose_list[6]
    return msg


def pose_to_list(pose):
    pose_list = []
    pose_list.append(pose.position.x)
    pose_list.append(pose.position.y)
    pose_list.append(pose.position.z)
    pose_list.append(pose.orientation.x)
    pose_list.append(pose.orientation.y)
    pose_list.append(pose.orientation.z)
    pose_list.append(pose.orientation.w)
    return pose_list

def pose_difference_np(pose, pose_ref):
    """
    Compute the approximate difference between two poses, by comparing
    the norm between the positions and using the quaternion difference
    to compute the rotation similarity

    Args:
        pose (list or np.ndarray): pose 1, in form [pos, ori], where
            pos (shape: [3,]) is of the form [x, y, z], and ori (shape: [4,])
            if of the form [x, y, z, w]
        pose_ref (list or np.ndarray): pose 2, in form [pos, ori], where
            pos (shape: [3,]) is of the form [x, y, z], and ori (shape: [4,])
            if of the form [x, y, z, w]

    Returns:
        2-element tuple containing:
        - np.ndarray: Euclidean distance between positions
        - np.ndarray: Quaternion difference between the orientations
    """
    if pose.ndim == 1:
        pose = pose[None, :]
    pos = pose[:, :3]
    pos_ref = pose_ref[:3]

    orientations = pose[:, 3:]
    ori_ref = pose_ref[3:]

    pos_diff = np.linalg.norm(pos - pos_ref, axis=1)

    rot_similarity_vec = np.zeros((orientations.shape[0],))
    for i, ori in enumerate(orientations):
        quat_diff = tf.transformations.quaternion_multiply(
            tf.transformations.quaternion_inverse(ori_ref),
            ori
        )

        rot_similarity = np.abs(quat_diff[3])
        rot_similarity_vec[i] = 1-rot_similarity
    # quat_diff = None  # TODO
    return pos_diff, rot_similarity_vec
# def pose_difference_np(pose, pose_ref, rs=False):
#     """
#     Compute the approximate difference between two poses, by comparing
#     the norm between the positions and using the quaternion difference
#     to compute the rotation similarity

#     Args:
#         pose (np.ndarray): pose 1, in form [pos, ori], where
#             pos (shape: [3,]) is of the form [x, y, z], and ori (shape: [4,])
#             if of the form [x, y, z, w]
#         pose_ref (np.ndarray): pose 2, in form [pos, ori], where
#             pos (shape: [3,]) is of the form [x, y, z], and ori (shape: [4,])
#             if of the form [x, y, z, w]
#         rs (bool): If True, use rotation_similarity metric for orientation error.
#             Otherwise use geodesic distance. Defaults to False

#     Returns:
#         2-element tuple containing:
#         - np.ndarray: Euclidean distance between positions
#         - np.ndarray: Quaternion difference between the orientations
#     """
#     pos_1, pos_2 = pose[:3], pose_ref[:3]
#     ori_1, ori_2 = pose[3:], pose_ref[3:]

#     pos_diff = pos_1 - pos_2
#     pos_error = np.linalg.norm(pos_diff)

#     quat_diff = quat_multiply(quat_inverse(ori_1), ori_2)
#     rot_similarity = np.abs(quat_diff[3])

#     # dot_prod = np.dot(ori_1, ori_2)
#     dot_prod = np.clip(np.dot(ori_1, ori_2), 0, 1)
#     angle_diff = np.arccos(2*dot_prod**2 - 1)

#     if rs:
#         angle_diff = 1 - rot_similarity
#     return pos_error, angle_diff


def pose_from_vectors(x_vec, y_vec, z_vec, trans, frame_id="yumi_body"):
    # Normalized frame
    hand_orient_norm = np.vstack((x_vec, y_vec, z_vec))
    hand_orient_norm = hand_orient_norm.transpose()
    quat = mat2quat(hand_orient_norm)
    # define hand pose
    pose = convert_pose_type(list(trans) + list(quat),
                             type_out="PoseStamped",
                             frame_out=frame_id)
    return pose


def transform_vectors(vectors, pose_transform):
    """Transform a set of vectors

    Args:
        vectors (np.ndarray): Numpy array of vectors, size
            [N, 3], where each row is a vector [x, y, z]
        pose_transform (PoseStamped): PoseStamped object defining the transform
    """
    vectors_homog = np.ones((4, vectors.shape[0]))
    vectors_homog[:-1, :] = vectors.T

    T_transform = matrix_from_pose(pose_transform)

    vectors_trans_homog = np.matmul(T_transform, vectors_homog)
    vectors_trans = vectors_trans_homog[:-1, :].T
    return vectors_trans

def sample_orthogonal_vector(reference_vector):
    """Sample a random unit vector that is orthogonal to the specified reference

    Args:
        reference_vector (np.ndarray): Numpy array with
            reference vector, [x, y, z]. Cannot be all zeros

    Return:
        np.ndarray: Size [3,] that is orthogonal to specified vector
    """
    y_unnorm = np.zeros(reference_vector.shape)

    nonzero_inds = np.where(reference_vector)[0]
    ind_1 = random.sample(nonzero_inds, 1)[0]
    while True:
        ind_2 = np.random.randint(3)
        if ind_1 != ind_2:
            break

    y_unnorm[ind_1] = reference_vector[ind_2]
    y_unnorm[ind_2] = -reference_vector[ind_1]
    y = y_unnorm / np.linalg.norm(y_unnorm)
    return y


def project_point2plane(point, plane_normal, plane_points):
    '''project a point to a plane'''
    point_plane = plane_points[0]
    w = point - point_plane
    dist = (np.dot(plane_normal, w) / np.linalg.norm(plane_normal))
    projected_point = point - dist * plane_normal / np.linalg.norm(plane_normal)
    return projected_point, dist


def body_world_yaw(current_pose, theta=None):
    """Given some initial pose, sample a new pose that is
    a pure yaw about the world frame orientation, with
    the origin at the current pose position

    Args:
        current_pose (PoseStamped): Current pose, to be yawed
        theta (float): Angle by which to yaw. If None, random
            theta in range [0, 2*pi] will be sampled

    Returns:
        PoseStamped: New pose, after in place yaw applied
    """
    current_pose_list = pose_stamped2list(current_pose)

    trans_to_origin = np.asarray(current_pose_list[:3])
    if theta is None:
        theta = np.random.random() * 2 * np.pi
    # yaw = R.from_euler('xyz', [0, 0, theta]).as_dcm()
    yaw = tf.transformations.euler_matrix(0, 0, theta, 'sxyz')

    # translate the source to the origin
    T_0 = np.eye(4)
    T_0[:-1, -1] = -trans_to_origin

    # apply pure rotation in the world frame
    T_1 = np.eye(4)
    T_1[:-1, :-1] = yaw

    # translate in [x, y] back away from origin
    T_2 = np.eye(4)
    T_2[0, -1] = trans_to_origin[0]
    T_2[1, -1] = trans_to_origin[1]
    T_2[2, -1] = trans_to_origin[2]
    yaw_trans = np.matmul(T_2, np.matmul(T_1, T_0))
    yaw_trans_pose = pose_from_matrix(yaw_trans)

    new_pose = transform_pose(current_pose, yaw_trans_pose)
    # new_pose_list = pose_stamped2list(new_pose)
    return new_pose


def rand_body_yaw_transform(pos, min_theta=0.0, max_theta=2*np.pi):
    """Given some initial position, sample a Transform that is
    a pure yaw about the world frame orientation, with
    the origin at the current pose position

    Args:
        pos (np.ndarray): Current position in the world frame
        min (float, optional): Minimum boundary for sample
        max (float, optional): Maximum boundary for sample

    Returns:
        np.ndarray: Transformation matrix
    """
    if isinstance(pos, list):
        pos = np.asarray(pos)
    trans_to_origin = pos
    theta = np.random.random() * (max_theta - min_theta) + min_theta
    # yaw = R.from_euler('xyz', [0, 0, theta]).as_dcm()
    yaw = tf.transformations.euler_matrix(0, 0, theta, 'sxyz')[:3, :3]

    # translate the source to the origin
    T_0 = np.eye(4)
    T_0[:-1, -1] = -trans_to_origin

    # apply pure rotation in the world frame
    T_1 = np.eye(4)
    T_1[:-1, :-1] = yaw

    # translate in [x, y] back away from origin
    T_2 = np.eye(4)
    T_2[0, -1] = trans_to_origin[0]
    T_2[1, -1] = trans_to_origin[1]
    T_2[2, -1] = trans_to_origin[2]
    yaw_trans = np.matmul(T_2, np.matmul(T_1, T_0))
    return yaw_trans


def get_base_pose_pb(obj_id, pb_client_id=0):
    pose = p.getBasePositionAndOrientation(obj_id, physicsClientId=pb_client_id)
    pos, ori = list(pose[0]), list(pose[1])
    pose = list2pose_stamped(pos + ori)
    return pose


def flip_palm_pulling(right_pose):
    # vecs = vec_from_pose(right_pose)
    # x_vec, y_vec, z_vec = vecs[0], vecs[1], vecs[2]

    # new_z_vec = z_vec
    # new_z_vec[1] = -1*z_vec
    # new_y_vec = y_vec
    # new_x_vec = np.cross(new_y_vec, new_z_vec)
    # l_pose = pose_from_vectors(new_x_vec, new_y_vec, new_z_vec, pose_stamped2list(right_pose)[:3])
    vecs = []
    for vec in vec_from_pose(right_pose):
        new_vec = vec
        new_vec[1] = -1*new_vec[1]
        vecs.append(new_vec)
    vecs[0] = np.cross(vecs[1], vecs[2])
    l_pose = pose_from_vectors(vecs[0], vecs[1], vecs[2], pose_stamped2list(right_pose)[:3])
    return l_pose


def interpolate_joint_trajectory(joint_trajectory, N):
    """Function to interpolate a joint trajectory so that it's more dense

    Args:
        joint_trajectory (np.ndarray): N x N_dof array of joints trajectory values
        N (int): Desired number of points in final joint trajectory. Cannot guarantee that the
            exact number will be reached, but it will be close.
    """
    dense_joint_trajectory = []
    s = joint_trajectory.shape
    n_per = N/s
    for i in range(joint_trajectory.shape[0] - 1):
        joints_current = joint_trajectory[i, :]
        joints_next = joint_trajectory[i+1, :]

        dense_joints = np.linspace(joints_current, joints_next, num=n_per)
        dense_joint_trajectory.append(dense_joints)
    return np.asarray(dense_joint_trajectory)