import sys, os
sys.path.append(os.environ['CODE_BASE'] + '/catkin_ws/src/config/src')
from helper import roshelper
import transformations as tfm
import numpy as np
from geometry_msgs.msg import PoseStamped

np.random.seed(0)
seed1 = np.random.random(1000000)
seed2 = np.random.random(1000000)
counter = 0

def arm_index(arm="r"):
    if arm=="r":
        return 1
    else:
        return 0

def point2line_distance(_p, _q, _rs):
    '''compute euclidean distance between a point and a line '''
    p = np.array(_p)
    q = np.array(_q)
    rs = np.array(_rs)
    x = p-q
    return np.linalg.norm(
        np.outer(np.dot(rs-q, x)/np.dot(x, x), x)+q-rs,
        axis=1)[0]

def normal_from_points(point_list, normal_guide=None):
    pt1 = point_list[0]
    pt2 = point_list[1]
    pt3 = point_list[2]
    v1 = np.array(pt2) - np.array(pt1)
    v2 = np.array(pt3) - np.array(pt1)
    v3 = np.cross(v1, v2)
    v3 = v3 / np.linalg.norm(v3)
    if normal_guide is not None:
        if np.dot(v3, normal_guide) < 0:
            v3 = -v3
    return v3

def sort_angle(val_dict):
    return val_dict['contact_angle']

def get2dpose_object(pose3d, object):
    #1. transform object to pose3d
    T_posed3d = roshelper.matrix_from_pose(pose3d)
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
    pose_rot_stable_placement = roshelper.pose_from_matrix(T_stable_placement)
    pose_transform = roshelper.get_transform(pose3d, pose_rot_stable_placement)
    T_pose_transform = roshelper.matrix_from_pose(pose_transform)
    euler = tfm.euler_from_matrix(T_pose_transform, 'rxyz')
    theta = euler[2]
    pose2d = np.array([x,y,theta])
    return pose2d, placement_id

def get3dpose_object(pose2d, object, stable_placement=0, frame_id="yumi_body"):
    mesh = object.stable_placement_dict['mesh'][stable_placement]
    volume, cog, inertia = mesh.get_mass_properties()
    # 1. get orientation pose of stable placement
    T_stable_placement = object.stable_placement_dict['T'][stable_placement]
    pose_stable_placement = roshelper.pose_from_matrix(T_stable_placement,
                                             frame_id=frame_id)

    #2. rotate by theta about the vertical axis
    T_pose_transform = tfm.euler_matrix(0, 0, pose2d[2], 'sxyz')
    pose_transform = roshelper.pose_from_matrix(T_pose_transform,
                                         frame_id=frame_id)
    pose_stable_placement_rotated = roshelper.transform_pose(pose_stable_placement,
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


def transform_point_list(face, T):
    """Perform a sequence of transformations T on a list of 3d vectors"""
    point_list = []
    for i in range(len(face)):
        rotated_face = np.matmul(T, np.append(face[i], 1))
        point_list.append(rotated_face[0:3])
    return point_list

def point_on_triangle(vertices):
    """get a random from within the boundaries of a triangle"""
    global counter
    A = vertices[0]
    B = vertices[1]
    C = vertices[2]
    r1 = seed1[counter]#random.uniform(0,1)
    r2 = seed2[counter]#random.uniform(0,1)
    counter += 1
    return (1 - np.sqrt(r1)) * A + (np.sqrt(r1) * (1 - (r2))) * B + (np.sqrt(r1) * r2) * C

def project_point2plane(point, plane_normal, plane_points):
    '''project a point to a plane'''
    point_plane = plane_points[0]
    w = point - point_plane
    dist = (np.dot(plane_normal, w) / np.linalg.norm(plane_normal))
    projected_point = point - dist * plane_normal / np.linalg.norm(plane_normal)
    return projected_point, dist

def point_in_triangle(query_point,
                      vertices):
    '''check if a point is within the boundaries of a triangle'''
    triangle_vertice_0 = vertices[0]
    triangle_vertice_1 = vertices[1]
    triangle_vertice_2 = vertices[2]

    u = triangle_vertice_1 - triangle_vertice_0
    v = triangle_vertice_2 - triangle_vertice_0
    n = np.cross(u, v)
    w = query_point - triangle_vertice_0
    gamma = np.dot(np.cross(u,w), n) / np.dot(n, n)
    beta = np.dot(np.cross(w,v), n) / np.dot(n, n)
    alpha = 1 - gamma - beta
    if 0 <= alpha and alpha <=1 and 0 <=beta and beta <=1 and 0 <= gamma and gamma <=1:
        return True
    else:
        return False
