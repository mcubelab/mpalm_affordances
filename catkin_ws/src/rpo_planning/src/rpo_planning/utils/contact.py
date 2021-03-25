import numpy as np
import open3d

from rpo_planning.utils import common as util

def correct_grasp_pos(contact_positions, pcd_pts):
    """Correct position of palms based on vector between
    the right and left palm positions

    Args:
        contact_positions (dict): Keyed 'right' and 'left', with
            values as [x, y, z] numpy arrays
        pcd_pts (np.ndarray): Points in pointcloud

    Returns:
        dict: Keyed 'right' and 'left', contains corrected palm positions
    """
    contact_world_frame_pred_r = contact_positions['right']
    contact_world_frame_pred_l = contact_positions['left']
    contact_world_frame_corrected = {}

    r2l_vector = contact_world_frame_pred_r - contact_world_frame_pred_l
    right_endpoint = contact_world_frame_pred_r + r2l_vector
    left_endpoint = contact_world_frame_pred_l - r2l_vector
    midpoint = contact_world_frame_pred_l + r2l_vector/2.0

    r_points_along_r2l = np.linspace(right_endpoint, midpoint, 200)
    l_points_along_r2l = np.linspace(midpoint, left_endpoint, 200)
    points = {}
    points['right'] = r_points_along_r2l
    points['left'] = l_points_along_r2l

    dists = {}
    dists['right'] = []
    dists['left'] = []

    inds = {}
    inds['right'] = []
    inds['left'] = []

    pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(np.concatenate(pcd_pts))
    pcd.points = open3d.utility.Vector3dVector(pcd_pts)

    kdtree = open3d.geometry.KDTreeFlann(pcd)
    for arm in ['right', 'left']:
        for i in range(points[arm].shape[0]):
            pos = points[arm][i, :]

            nearest_pt_ind = kdtree.search_knn_vector_3d(pos, 1)[1][0]

#             dist = (np.asarray(pcd.points)[nearest_pt_ind] - pos).dot(np.asarray(pcd.points)[nearest_pt_ind] - pos)
            dist = np.asarray(pcd.points)[nearest_pt_ind] - pos

            inds[arm].append((i, nearest_pt_ind))
            dists[arm].append(dist.dot(dist))

    for arm in ['right', 'left']:
        min_ind = np.argmin(dists[arm])
#         print(min_ind)
#         print(len(inds[arm]))
        min_point_ind = inds[arm][min_ind][0]
#         nearest_pt_world = np.asarray(pcd.points)[min_point_ind]
        nearest_pt_world = points[arm][min_point_ind]
        contact_world_frame_corrected[arm] = nearest_pt_world

    return contact_world_frame_corrected


def correct_palm_pos_single(contact_pose, pcd_pts):
    """Correct position of palms based on vector between
    the right and left palm positions

    Args:
        contact_pose (np.ndarray): World frame [x, y, z, x, y, z, w]
        pcd_pts (np.ndarray): Points in pointcloud

    Returns:
        np.ndarray: corrected palm position
    """
    # compute center of mass, to know if we should move in or out
    center_of_mass = np.mean(pcd_pts, axis=0)

    # compute closest point to the position
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pcd_pts)
    kdtree = open3d.geometry.KDTreeFlann(pcd)
    nearest_pt_ind = kdtree.search_knn_vector_3d(contact_pose[:3], 1)[1][0]
    
    vec_to_point = np.asarray(pcd.points)[nearest_pt_ind] - contact_pose[:3]
    vec_to_com = center_of_mass - contact_pose[:3]

    # check if these vectors point in the same direciton with dot product
    dot_prod = np.dot(vec_to_point/np.linalg.norm(vec_to_point), vec_to_com/np.linalg.norm(vec_to_com))
    contact_is_outside_object = dot_prod > np.cos(np.deg2rad(85))
    if contact_is_outside_object:
        # print('outside object!')
        new_pos = np.asarray(pcd.points)[nearest_pt_ind]
        new_pose = new_pos.tolist() + contact_pose[3:].tolist()
        return np.asarray(new_pose)

    # print('inside object')
    # get y vector
    normal_y = util.list2pose_stamped([0, 1, 0, 0, 0, 0, 1])
    normal_y_pose_world = util.pose_stamped2np(util.transform_pose(normal_y, util.list2pose_stamped(contact_pose)))
    world_frame_y_vec = normal_y_pose_world[:3] - contact_pose[:3]

    # compute vectors and points going away from CoM
    # endpoint_1 = center_of_mass
    # endpoint_2 = center_of_mass + world_frame_y_vec * 1.0
    endpoint_1 = contact_pose[:3]
    endpoint_2 = endpoint_1 + world_frame_y_vec * 1.0    
    points_along_y_vec = np.linspace(endpoint_1, endpoint_2, 500)

    dists = []
    inds = []

    for i in range(points_along_y_vec.shape[0]):
        pos = points_along_y_vec[i, :]

        nearest_pt_ind = kdtree.search_knn_vector_3d(pos, 1)[1][0]

#             dist = (np.asarray(pcd.points)[nearest_pt_ind] - pos).dot(np.asarray(pcd.points)[nearest_pt_ind] - pos)
        dist = np.asarray(pcd.points)[nearest_pt_ind] - pos

        inds.append((i, nearest_pt_ind))
        dists.append(dist.dot(dist))

    min_ind = np.argmin(dists)
    min_point_ind = inds[min_ind][0]
    nearest_pt_world = points_along_y_vec[min_point_ind]
    contact_pos_world_frame_corrected = nearest_pt_world.tolist()
    return np.asarray(contact_pos_world_frame_corrected + contact_pose[3:].tolist())


def project_point2plane(point, plane_normal, plane_points):
    '''project a point to a plane'''
    point_plane = plane_points[0]
    w = point - point_plane
    dist = (np.dot(plane_normal, w) / np.linalg.norm(plane_normal))
    projected_point = point - dist * plane_normal / np.linalg.norm(plane_normal)
    return projected_point, dist