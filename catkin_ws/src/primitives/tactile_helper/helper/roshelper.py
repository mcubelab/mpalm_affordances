import numpy as np
import tf, rospy
import subprocess, sys, os
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from tf.transformations import quaternion_from_euler, quaternion_from_matrix
import tf.transformations as tfm
from copy import deepcopy
import helper
import rospy
import visualize_helper
import sys, os
sys.path.append(os.environ["CODE_BASE"] + "/catkin_ws/src/tactile_dexterity/src")

def terminate_ros_node(s):
    list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
    list_output = list_cmd.stdout.read()
    retcode = list_cmd.wait()
    assert retcode == 0, "List command returned %d" % retcode
    for term in list_output.split("\n"):
        if (term.startswith(s)):
            os.system("rosnode kill " + term)
            print "rosnode kill " + term

def start_ros_node(node, package = 'PlanarManipulation'):
    os.system("rosrun " + package + node )

def set_joints_virtual(plan):
    # for counter in range(1000):
    publish_yumi_joints(
        helper.deorder_joints(plan['joints'][1]), "r")
    publish_yumi_joints(
        helper.deorder_joints(plan['joints'][0]), "l")
    # joints_states = rospy.wait_for_message("joint_states", JointState)
    des_joints = [0] * 14
    des_joints[0:7] = plan['joints'][1]
    des_joints[7:14] = plan['joints'][0]

def sort_face_lines(face_points, poses_tip_world, arm='l'):
    """
    For a given object face (defined as 4 points) -> sort the lines in terms of which is closest to robot end effector
    :param face_points:
    :param poses_tip_world:
    :param arm:
    :return:
    """
    arm_id = arm_index(arm)
    line_list = []
    dist_list = []
    for counter, point in enumerate(face_points):
        if counter==len(face_points)-1:
            line_list.append([face_points[counter], face_points[0]])
        else:
            line_list.append([face_points[counter], face_points[counter+1]])
        tip_position = [poses_tip_world[arm_id].pose.position.x,
                            poses_tip_world[arm_id].pose.position.y,
                            poses_tip_world[arm_id].pose.position.z,]
        dist_list.append(helper.point2line_distance(line_list[counter][0], line_list[counter][1], tip_position))
    index = np.argmin(np.array(dist_list))
    sorted_dist_list = np.sort(dist_list)
    arg_sorted_dist_list = np.argsort(dist_list)
    #if almost equal to 2 lines, pick the one that is closest to robot (to make apriltag visible)
    if abs(sorted_dist_list[0] - sorted_dist_list[1]) < 0.001:
        if (line_list[arg_sorted_dist_list[0]][0][0] + line_list[arg_sorted_dist_list[0]][1][0]) /2 > (line_list[arg_sorted_dist_list[1]][0][0] + line_list[arg_sorted_dist_list[1]][1][0]) /2:
            tmp = deepcopy(arg_sorted_dist_list[0])
            arg_sorted_dist_list[0] = deepcopy(arg_sorted_dist_list[1])
            arg_sorted_dist_list[1] = deepcopy(tmp)
            # index = arg_sorted_dist_list[1]
    index_list = [arg_sorted_dist_list[0], arg_sorted_dist_list[1]]
    return line_list, index_list



def transform_stable_config(face_list_proposals_base, normal_list_proposals_base, pose_proposals_base):
    # 1. convert face points from proposals base to world frame
    points_world_list = []
    for points in face_list_proposals_base:
        points_world_list_tmp = []
        for point in points:
            msg_proposals_base = list2pose_stamped([point[0],
                                                      point[1],
                                                      point[2],
                                                      0, 0, 0, 1])
            msg_world = convert_reference_frame(msg_proposals_base,
                                                          unit_pose(),
                                                          pose_proposals_base,
                                                          "yumi_body")
            points_world_list_tmp.append(np.array([msg_world.pose.position.x,
                                      msg_world.pose.position.y,
                                      msg_world.pose.position.z, ]))
        points_world_list.append(points_world_list_tmp)
        # 2. convert normals from proposals base to world frame
        normals_world_list = []
        for normal in normal_list_proposals_base:
            T = matrix_from_pose(pose_proposals_base)
            R = T[0:3, 0:3]
            normal_world = np.matmul(R, normal)
            normals_world_list.append(normal_world)
    return points_world_list, normals_world_list

def convert_gripper_to_tip_frame(gripper_poses):
    #compute location of end-effector
    tip_to_wrist = list2pose_stamped(rospy.get_param('transform/tip_to_wrist'))
    poses_tip_world = [convert_reference_frame(tip_to_wrist,
                                                   unit_pose(),
                                                   gripper_poses[0],
                                                   "yumi_body"),
                      convert_reference_frame(tip_to_wrist,
                                                        unit_pose(),
                                                        gripper_poses[1],
                                                        "yumi_body")]
    return poses_tip_world

def reorder_lines(points_list):
    points_array  = np.array(points_list)
    if all(abs(points_list[i][0]-points_list[0][0]) < 1e-5 for i in range(len(points_list))):
        index = [1,2]
        constant_index = 0
    elif all(abs(points_list[i][1]-points_list[0][1]) < 1e-5 for i in range(len(points_list))):
        index = [0,2]
        constant_index = 1
    else:
        index = [0,1]
        constant_index = 2

    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    points2d = points_array[:,index]
    hull = ConvexHull(points2d)
    ordered_point_array = np.zeros((len(points_list), 3))
    ordered_point_array[:,index[0]] = points2d[hull.vertices,0]
    ordered_point_array[:,index[1]] = points2d[hull.vertices,1]
    ordered_point_array[:,constant_index] = np.array(points_list)[:, constant_index]

    return list(ordered_point_array)


def get_object_convex_face_id(poses_tip_world, object, stable_placement, pose_proposals_base, arm="r"):
    arm_id = arm_index(arm)
    #1. initialize values
    pose_tip_world = poses_tip_world[arm_id]
    face_list_proposals_base = object.stable_placement_dict['convex_face_stable_config'][stable_placement]
    normal_list_proposals_base = object.stable_placement_dict['normal_stable_config'][stable_placement]
    face_list_world, normal_list_world = transform_stable_config(face_list_proposals_base,
                                                                 normal_list_proposals_base,
                                                                 pose_proposals_base)
    #2. extract gripper y_axis in world frame
    T_pose_tip_world =  matrix_from_pose(pose_tip_world)
    y_axis = T_pose_tip_world[0:3,1]

    #3. find angle between y_axis of gripper and all faces
    dict_faces = {}
    face_id_list = []
    face_list = []
    dist_to_gripper_list = []
    mid_point_list = []
    for counter, normal in enumerate(normal_list_world):
        projection = np.linalg.norm(np.dot(y_axis/np.linalg.norm(y_axis), normal/np.linalg.norm(normal)))
        if projection > 0.8:
            total_points = 0
            for points in face_list_world[counter]:
                total_points += points
            mid_point = total_points / len(face_list_world[counter])
            face_id_list.append(counter)
            face_list.append(face_list_world[counter])
            mid_point_list.append(mid_point)
            tip_position = np.array([poses_tip_world[arm_id].pose.position.x,
                                     poses_tip_world[arm_id].pose.position.y,
                                     poses_tip_world[arm_id].pose.position.z])
            dist_to_gripper_list.append(np.linalg.norm(np.array(mid_point) - tip_position))
    dict_faces['face_id'] = face_id_list
    dict_faces['dist_to_gripper_list'] = dist_to_gripper_list
    dict_faces['mid_point_list'] = mid_point_list
    dict_faces['faces'] = face_list
    index = np.argmin(dist_to_gripper_list)
    dict_faces['faces_proposals_base'] = face_list_proposals_base[index]
    return index, dict_faces

def arm_index(arm="r"):
    if arm=="r":
        return 1
    else:
        return 0


def set_cart_virtual(plan):
    for counter in range(10):
        visualize_helper.visualize_object(pose_left,
                                          filepath="package://yumi_description/meshes/mpalm/mpalms_all_coarse.stl",
                                          name="/gripper_left",
                                          color=(0., 0., 1., 1.),
                                          frame_id="/yumi_body")

        visualize_helper.visualize_object(pose_right,
                                          filepath="package://yumi_description/meshes/mpalm/mpalms_all_coarse.stl",
                                          name="/gripper_right",
                                          color=(0., 0., 1., 1.),
                                          frame_id="/yumi_body")


def publish_float(pub, value):
    js = Float64()
    js.data = value
    pub.publish(js)

def publish_object_pose_2d(br, topic_name, pose):
    object_pos_pub = rospy.Publisher(topic_name, Pose, queue_size=10)
    quaternion = quaternion_from_euler(0, 0, pose[2])
    msg = Pose()
    msg.position.x = pose[0]
    msg.position.y = pose[1]
    msg.position.z = 0
    msg.orientation.x = quaternion[0]
    msg.orientation.y = quaternion[1]
    msg.orientation.z = quaternion[2]
    msg.orientation.w = quaternion[3]
    #hack: publish twice to give it time
    for i in range(2):
        rospy.sleep(.1)
        handle_block_pose(msg, br, 'yumi_body', 'object2')

def publish_object_pose(br, topic_name, pose):
    object_pos_pub = rospy.Publisher(topic_name, Pose, queue_size=10)
    msg = pose
    #hack: publish twice to give it time
    for i in range(2):
        rospy.sleep(.1)
        handle_block_pose(msg, br, 'yumi_body', 'object2')

def mat2quat(orient_mat_3x3):
    orient_mat_4x4 = [[orient_mat_3x3[0][0],orient_mat_3x3[0][1],orient_mat_3x3[0][2],0],
                       [orient_mat_3x3[1][0],orient_mat_3x3[1][1],orient_mat_3x3[1][2],0],
                       [orient_mat_3x3[2][0],orient_mat_3x3[2][1],orient_mat_3x3[2][2],0],
                       [0,0,0,1]]

    orient_mat_4x4 = np.array(orient_mat_4x4)
    quat = quaternion_from_matrix(orient_mat_4x4)
    return quat

def handle_block_pose(msg, br, base_frame, target_frame):
    for i in range(3):
        br.sendTransform((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                         (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w),
                         rospy.Time.now(),
                         target_frame,
                         base_frame)

def publish_yumi_joints(joints, arm_name):
    if arm_name=="r":
        jnames = ['yumi_joint_1_r',
              'yumi_joint_2_r',
              'yumi_joint_3_r',
              'yumi_joint_4_r',
              'yumi_joint_5_r',
              'yumi_joint_6_r',
              'yumi_joint_7_r']
    else:
        jnames = ['yumi_joint_1_l',
                  'yumi_joint_2_l',
                  'yumi_joint_3_l',
                  'yumi_joint_4_l',
                  'yumi_joint_5_l',
                  'yumi_joint_6_l',
                  'yumi_joint_7_l']

    name = "/yumi/joint_states"
    js = JointState()
    js.name = jnames
    js.position = [joints[x] for x in range(7)]
    joint_states_pub = rospy.Publisher(name, JointState, queue_size=10)
    joint_states_pub.publish(js)


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
        pose_interp.append(convert_pose_type(pose_tmp, type_out="PoseStamped", frame_out=frame_id))
    return pose_interp

def interpolate_pose_discrete(pose_initial, pose_final, frac):
    frame_id = pose_initial.header.frame_id
    pose_initial_list = pose_stamped2list(pose_initial)
    pose_final_list = pose_stamped2list(pose_final)
    trans_initial = pose_initial_list[0:3]
    quat_initial = pose_initial_list[3:7]
     # onvert to pyquaterion convertion (w,x,y,z)
    trans_final = pose_final_list[0:3]
    quat_final = pose_final_list[3:7]

    trans_interp = [np.interp(frac, [0, 1], [trans_initial[0], trans_final[0]]),
                         np.interp(frac, [0, 1], [trans_initial[1], trans_final[1]]),
                         np.interp(frac, [0, 1], [trans_initial[2], trans_final[2]]),
                         ]

    quat_interp = tf.transformations.quaternion_slerp(quat_initial,
                                                      quat_final,
                                                      frac)
    pose_interp = [trans_interp[0],
                trans_interp[1],
                trans_interp[2],
                quat_interp[0], #return in ROS ordering w,x,y,z
                quat_interp[1],
                quat_interp[2],
                quat_interp[3],
                ]
    return convert_pose_type(pose_interp, type_out="PoseStamped", frame_out=frame_id)

def convert_pose_type(pose, type_out="list", frame_out="yumi_body"):
    type_in = type(pose)
    assert (type_in in [Pose, list, np.ndarray, PoseStamped, dict])
    assert (type_out in ["Pose", "list", "ndarray", "PoseStamped", "dict"])
    #convert all poses to list
    if type_in == Pose:
        pose = pose2list(pose)
    elif type_in == PoseStamped:
        pose = pose_stamped2list(pose)
    elif type_in==dict:
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

def pose2list(msg):
    return [float(msg.position.x),
            float(msg.position.y),
            float(msg.position.z),
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
            ]
def pose_stamped2list(msg):
    return [float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
            ]

def matrix_from_pose(pose, type="PoseStamped"):
    assert (type in ["Pose", "list", "ndarray", "PoseStamped"])
    if type=="Pose":
        pose = pose2list(pose)
    elif type=="PoseStamped":
        pose = pose_stamped2list(pose)
    trans = pose[0:3]
    quat = pose[3:7]
    T = tf.transformations.quaternion_matrix(quat)
    T[0:3,3] = trans
    return T

def pose_from_matrix(matrix, type="PoseStamped", frame_id="yumi_body"):
    assert (type in ["Pose", "list", "ndarray", "PoseStamped"])
    trans = tf.transformations.translation_from_matrix(matrix)
    quat = tf.transformations.quaternion_from_matrix(matrix)
    pose = list(trans) + list(quat)
    if type=="Pose":
        pose = list2pose(pose)
    elif type=="PoseStamped":
        pose = list2pose_stamped(pose, frame_id=frame_id)
    elif type=="list":
        pose = pose
    elif type=="ndarray":
        pose = np.array(pose)
    return pose

def list2pose_stamped(pose, frame_id="proposals_base"):
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

def point2pose_stamped(point, frame_id="yumi_body"):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = point[0]
    msg.pose.position.y = point[1]
    msg.pose.position.z = point[2]
    msg.pose.orientation.x = 0
    msg.pose.orientation.y = 0
    msg.pose.orientation.z = 0
    msg.pose.orientation.w = 1
    return msg

def pose_stamped2point(msg):
    return np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

def dict2pose_stamped(dict):
    msg = PoseStamped()
    msg.header.frame_id = dict['header']['frame_id']
    msg.pose.position.x = dict['pose']['position']['x']
    msg.pose.position.y = dict['pose']['position']['y']
    msg.pose.position.z = dict['pose']['position']['z']
    msg.pose.orientation.x = dict['pose']['orientation']['x']
    msg.pose.orientation.y = dict['pose']['orientation']['y']
    msg.pose.orientation.z = dict['pose']['orientation']['z']
    msg.pose.orientation.w = dict['pose']['orientation']['w']
    return msg

def list2dict(pose, frame_id):
    msg = list2pose_stamped(pose, frame_id)
    return pose_stamped2dict(msg)

def pose_stamped2dict(msg):
    pose_dict = {}
    pose_dict['header'] = {}
    pose_dict['pose'] = {}
    pose_dict['pose']['position'] = {}
    pose_dict['pose']['orientation'] = {}
    pose_dict['header']['frame_id'] = msg.header.frame_id
    pose_dict['pose']['position']['x'] = float(msg.pose.position.x)
    pose_dict['pose']['position']['y'] = float(msg.pose.position.y)
    pose_dict['pose']['position']['z'] = float(msg.pose.position.z)
    pose_dict['pose']['orientation']['x'] = msg.pose.orientation.x
    pose_dict['pose']['orientation']['y'] = msg.pose.orientation.y
    pose_dict['pose']['orientation']['z'] = msg.pose.orientation.z
    pose_dict['pose']['orientation']['w'] = msg.pose.orientation.w
    return pose_dict

def list2pose(pose):
    msg = Pose()
    msg.position.x = pose[0]
    msg.position.y = pose[1]
    msg.position.z = pose[2]
    msg.orientation.x = pose[3]
    msg.orientation.y = pose[4]
    msg.orientation.z = pose[5]
    msg.orientation.w = pose[6]
    return msg

def get3dpose_object(pose2d, object, stable_placement=0, frame_id="yumi_body"):
    mesh = object.stable_placement_dict['mesh'][stable_placement]
    volume, cog, inertia = mesh.get_mass_properties()
    # 1. get orientation pose of stable placement
    T_stable_placement = object.stable_placement_dict['T'][stable_placement]
    pose_stable_placement = pose_from_matrix(T_stable_placement,
                                             type="PoseStamped",
                                             frame_id=frame_id)

    #2. rotate by theta about the vertical axis
    T_pose_transform = tfm.euler_matrix(0, 0, pose2d[2], 'sxyz')
    pose_transform = pose_from_matrix(T_pose_transform,
                                         type="PoseStamped",
                                         frame_id=frame_id)
    pose_stable_placement_rotated = transform_pose(pose_stable_placement,
                   pose_transform)
    #3. convert to quaternion
    # quaternion = tfm.quaternion_from_euler(0, 0, pose2d[2])
    msg = PoseStamped()
    msg.pose.position.x = pose2d[0]
    msg.pose.position.y = pose2d[1]
    msg.pose.position.z = cog[2] - mesh.min_[2]
    msg.pose.orientation = pose_stable_placement_rotated.pose.orientation
    msg.header.frame_id = frame_id
    return msg

def get2dpose_object(pose3d, object):
    #1. transform object to pose3d
    T_posed3d = matrix_from_pose(pose3d)
    R = T_posed3d[0:3,0:3]
    #2. find placement with z=0
    vector_rotated_list = []
    for placement, vector in enumerate(object.stable_placement_dict['vector']):
        vector_rotated_list.append(np.matmul(R, vector)[2])

    placement_id = np.argmin(np.array(vector_rotated_list))
    #3. extract x and y of object
    x = pose3d.pose.position.x
    y = pose3d.pose.position.y
    #4. find rotation (about z axis) between stable_placement and posed3d orientation
    T_stable_placement = object.stable_placement_dict['T_rot'][placement_id]
    pose_rot_stable_placement = pose_from_matrix(T_stable_placement)
    pose_transform = get_transform(pose3d, pose_rot_stable_placement)
    T_pose_transform = matrix_from_pose(pose_transform)
    euler = tfm.euler_from_matrix(T_pose_transform, 'rxyz')
    theta = euler[2]
    pose2d = np.array([x,y,theta])
    return pose2d, placement_id

def line_push_obj2robot(pose2d, rbpb, side_dict):
    Cbi = helper.C3_2d(pose2d[2])
    R = helper.C3(pose2d[2]).transpose()
    ripb = np.matmul(np.transpose(Cbi), rbpb)
    ribi = pose2d[0:2]
    ripi = ribi + ripb
    hand_frame_b = side_dict['edge_frame_b']
    hand_orient_norm = np.vstack((np.matmul(R, hand_frame_b[0]),
                                  np.matmul(R, hand_frame_b[1]),
                                  np.matmul(R, hand_frame_b[2])))
    hand_orient_norm = hand_orient_norm.transpose()
    quaternion = mat2quat(hand_orient_norm)
    msg = PoseStamped()
    msg.pose.position.x = ripi[0]
    msg.pose.position.y = ripi[1]
    msg.pose.position.z = 0.045
    msg.pose.orientation.x = quaternion[0]
    msg.pose.orientation.y = quaternion[1]
    msg.pose.orientation.z = quaternion[2]
    msg.pose.orientation.w = quaternion[3]
    return msg

def proposals_base_from_object_pose(object, object_pose_world):
    object_pose_initial_2d, placement_object_2d = get2dpose_object(object_pose_world, object)
    T_proposal_base = tfm.euler_matrix(0, 0, object_pose_initial_2d[2], 'rxyz')
    pose_proposal_base = pose_from_matrix(T_proposal_base)
    msg_proposals = list2pose_stamped([object_pose_world.pose.position.x,
                                             object_pose_world.pose.position.y,
                                             0,
                                             pose_proposal_base.pose.orientation.x,
                                             pose_proposal_base.pose.orientation.y,
                                             pose_proposal_base.pose.orientation.z,
                                             pose_proposal_base.pose.orientation.w],
                                            frame_id="yumi_body")
    return msg_proposals

# def transform_points_list_to_proposals_base(line_points_initial, pose_proposals_base):
#     """
#     Converts list of poses from world frame to proposal base
#     :param line_points_initial:
#     :param pose_proposals_base:
#     :return:
#     """
#     line_points_proposals = []
#     for pose_lines_world in line_points_initial:
#         pose_lines_proposals_base = convert_reference_frame(pose_lines_world,
#                                                         pose_proposals_base,
#                                                         unit_pose(),
#                                                         "proposals_base")
#         line_points_proposals.append(pose_lines_proposals_base)
#     return line_points_proposals
#

def pull_obj2robot(pose, angle, delta_spring, object):
    mesh, vertices = object.transform_object(pose, type="PoseStamped")
    pose2d, placement = get2dpose_object(pose, object)
    ripi = pose2d[0:2]
    #1. define nominal orientation of gripper for pullin
    T_gripper_nominal_world = tfm.euler_matrix(0, np.pi/2, np.pi/2, 'rxyz')
    #2. rotate gripper by desired angle of gripper(sampling different solutions) + orientation of object about z axis
    T_transform = tfm.euler_matrix(0, 0, angle  + pose2d[2], 'rxyz')
    gripper_nominal_pose_world = pose_from_matrix(T_gripper_nominal_world, "PoseStamped", "yumi_body")
    pose_transform = pose_from_matrix(T_transform, "PoseStamped", "yumi_body")
    gripper_final_pose_world = transform_pose(gripper_nominal_pose_world, pose_transform)
    gripper_final_pose_world.pose.position.x = ripi[0]
    gripper_final_pose_world.pose.position.y = ripi[1]
    gripper_final_pose_world.pose.position.z = mesh.max_[2] - mesh.min_[2] + delta_spring
    #3. move gripper closer to edge (to get tactile imprint)
    gripper_poses = [gripper_final_pose_world, gripper_final_pose_world]
    poses_tip_world = deepcopy(gripper_poses)
    pose2d, stable_placement = get2dpose_object(pose, object)
    pose_proposals_base = proposals_base_from_object_pose(object, pose)
    pose_transform_frameB = unit_pose()
    if placement==0:
        pose_transform_frameB.pose.position.x -= 0.09 / 2
    elif placement==2:
        pose_transform_frameB.pose.position.z -= 0.02
    elif placement == 3:
        pose_transform_frameB.pose.position.z += 0.03
    elif placement == 1:
        pose_transform_frameB.pose.position.x += 0.02
        pose_transform_frameB.pose.position.z += 0.0
    elif placement == 4:
        pose_transform_frameB.pose.position.z += 0.02
    gripper_final_pose_world = transform_pose_intermediate_frame(gripper_final_pose_world,
                                                                pose,
                                                                pose_transform_frameB)

    #4 pull at corner
    # face_id, dict_faces = get_object_convex_face_id(gripper_poses,
    #                                                 object,
    #                                                 stable_placement,
    #                                                 pose_proposals_base,
    #                                                 arm="r")
    # corner_point_list = dict_faces['faces'][face_id]
    # gripper_final_pose_world.pose.position.x = corner_point_list[0][0]
    # gripper_final_pose_world.pose.position.y = corner_point_list[0][1]

    return gripper_final_pose_world

def get_transform(pose_frame_target, pose_frame_source):
    """
    Find transform that transforms pose source to pose target
    :param pose_frame_target:
    :param pose_frame_source:
    :return:
    """
    #both poses must be expressed in same reference frame
    T_target_world = matrix_from_pose(pose_frame_target, type="PoseStamped")
    T_source_world = matrix_from_pose(pose_frame_source, type="PoseStamped")
    T_relative_world = np.matmul(T_target_world, np.linalg.inv(T_source_world))
    pose_relative_world = pose_from_matrix(T_relative_world, type="PoseStamped", frame_id=pose_frame_source.header.frame_id)
    return pose_relative_world

def transform_pose(pose_source, pose_transform):
    T_pose_source = matrix_from_pose(pose_source, type="PoseStamped")
    T_transform_source = matrix_from_pose(pose_transform, type="PoseStamped")
    T_pose_final_source = np.matmul(T_transform_source, T_pose_source)
    pose_final_source = pose_from_matrix(T_pose_final_source, "PoseStamped", frame_id=pose_source.header.frame_id)
    return pose_final_source

def set_position(pose, position):
        pose_new = deepcopy(pose)
        pose_new.pose.position.x = position[0]
        pose_new.pose.position.y = position[1]
        pose_new.pose.position.z = position[2]
        return pose_new

def get_pos(pose):
        return np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])

def transform_pose_intermediate_frame(pose_source_frameA, pose_frameB, pose_transform_frameB):
    pose_source_frameB = convert_reference_frame(pose_source_frameA,
                                                  pose_frameB,
                                                  unit_pose(),
                                                  frame_id="frameB")
    pose_source_transformed_frameB = transform_pose(pose_source_frameB,
                                                    pose_transform_frameB)
    pose_source_transformed_frameA = convert_reference_frame(pose_source_transformed_frameB,
                                                  unit_pose(),
                                                  pose_frameB,
                                                  frame_id="frameB")
    return pose_source_transformed_frameA

def transform_pose_intermediate_frame_list(pose_source_frameA_list, pose_frameB, pose_transform_frameB):
    pose_source_transformed_frameA_list = []
    for pose_source_frameA in pose_source_frameA_list:
        pose_source_frameB = convert_reference_frame(pose_source_frameA,
                                                      pose_frameB,
                                                      unit_pose(),
                                                      frame_id="frameB")
        pose_source_transformed_frameB = transform_pose(pose_source_frameB,
                                                        pose_transform_frameB)
        pose_source_transformed_frameA = convert_reference_frame(pose_source_transformed_frameB,
                                                      unit_pose(),
                                                      pose_frameB,
                                                      frame_id="frameB")
        pose_source_transformed_frameA_list.append(pose_source_transformed_frameA)
    return pose_source_transformed_frameA_list

def convert_reference_frame_list(pose_source_list, pose_frame_target, pose_frame_source, frame_id = "yumi_body"):
    pose_target_list = []
    for pose_source in pose_source_list:
        pose_target_list.append(convert_reference_frame(pose_source,
                                                        pose_frame_target,
                                                        pose_frame_source,
                                                        frame_id))
    return pose_target_list

def convert_reference_frame(pose_source, pose_frame_target, pose_frame_source, frame_id = "yumi_body"):
    """
    Converts pose source from source frame to target frame
    :param pose_source:
    :param pose_frame_target:
    :param pose_frame_source:
    :param frame_id:
    :return:
    """
    T_pose_source = matrix_from_pose(pose_source, type="PoseStamped")
    pose_transform_target2source = get_transform(pose_frame_source, pose_frame_target)
    T_pose_transform_target2source = matrix_from_pose(pose_transform_target2source, type="PoseStamped")
    T_pose_target = np.matmul(T_pose_transform_target2source, T_pose_source)
    pose_target = pose_from_matrix(T_pose_target, "PoseStamped", frame_id=frame_id)
    return pose_target

def convert_list_reference_frames(pose_source_list, pose_frame_target_list, pose_frame_source_list, frame_id_list = "yumi_body"):
    pose_target_list = []
    for pose_source, pose_frame_target, pose_frame_source, frame_id in zip(pose_source_list, pose_frame_target_list, pose_frame_source_list, frame_id_list):
        T_pose_source = matrix_from_pose(pose_source, type="PoseStamped")
        pose_transform_target2source = get_transform(pose_frame_source, pose_frame_target)
        T_pose_transform_target2source = matrix_from_pose(pose_transform_target2source, type="PoseStamped")
        T_pose_target = np.matmul(T_pose_transform_target2source, T_pose_source)
        pose_target = pose_from_matrix(T_pose_target, "PoseStamped", frame_id=frame_id)
        pose_target_list.append(pose_target)
    return pose_target_list

def get_pose_from_tf_frame(listener, target_frame, source_name):
    trans, quat = listener.lookupTransform(source_name, target_frame, rospy.Time(0))
    pose_target_source = convert_pose_type(trans+quat, type_out="PoseStamped", frame_out=source_name)
    return pose_target_source

def transform_body(pose_source_world, pose_transform_target_body):
    #convert source to target frame
    pose_source_body = convert_reference_frame(pose_source_world,
                                                 pose_source_world,
                                                 unit_pose("yumi_body"),
                                                 frame_id="body_frame")
    #perform transformation in body frame
    pose_source_rotated_body = transform_pose(pose_source_body,
                                              pose_transform_target_body)
    # rotate back
    pose_source_rotated_world = convert_reference_frame(pose_source_rotated_body,
                                                         unit_pose("yumi_body"),
                                                         pose_source_world,
                                                         frame_id="yumi_body")
    return pose_source_rotated_world

def unit_pose(frame_id="yumi_body"):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.orientation.w = 1
    return msg

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

def vec_from_pose(pose):
    quat = pose.pose.orientation
    R = tfm.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
    x_vec = R[0:3, 0]
    y_vec = R[0:3, 1]
    x_vec = R[0:3, 2]
    return x_vec, y_vec, x_vec

def flip_orientation(pose, flip_axis='y', constant_axis='z'):
    '''set orientation of right gripper as a mirror reflection of left gripper about y-axis'''
    trans = [pose.pose.position.x,
             pose.pose.position.y,
             pose.pose.position.z]
    frame_id = pose.header.frame_id
    quat = pose.pose.orientation
    R = tfm.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
    if flip_axis=="x":
        if constant_axis=="y":
            x_vec = -R[0:3, 0]
            y_vec = R[0:3, 1]
            z_vec = np.cross(x_vec, y_vec)
        elif constant_axis=="z":
            x_vec = -R[0:3, 0]
            z_vec = R[0:3, 2]
            y_vec = np.cross(z_vec, x_vec)
    elif flip_axis=="y":
        if constant_axis == "x":
            x_vec = R[0:3, 0]
            y_vec = -R[0:3, 1]
            z_vec = np.cross(x_vec, y_vec)
        elif constant_axis == "z":
            z_vec = R[0:3, 2]
            y_vec = -R[0:3, 1]
            x_vec = np.cross(y_vec, z_vec)
    elif flip_axis == "z":
        if constant_axis == "x":
            z_vec = -R[0:3, 2]
            x_vec = R[0:3, 0]
            y_vec = -np.cross(x_vec, z_vec)
        elif constant_axis == "y":
            z_vec = -R[0:3, 2]
            y_vec = R[0:3, 1]
            x_vec = np.cross(y_vec, z_vec)
    else:
        return pose
    x_vec = x_vec / np.linalg.norm(x_vec)
    y_vec = y_vec / np.linalg.norm(y_vec)
    z_vec = z_vec / np.linalg.norm(z_vec)
    return pose_from_vectors(x_vec, y_vec, z_vec, trans, frame_id=frame_id)
