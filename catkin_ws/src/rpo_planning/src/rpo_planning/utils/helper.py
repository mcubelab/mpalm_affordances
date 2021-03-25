import numpy as np
import math


def angle_2d(vec):
    """get heading of 2d vector"""
    angle = np.arctan2(vec[1], vec[0])
    return angle


def vector_transform(T, r):
    """Perform 4x4 transformation matrix T to 3d vector r"""
    r_new = np.matmul(T, np.concatenate((r, np.array([1]))))
    return r_new[0:3]  


def elongate_vector(points, dist):
    """create a line of dist from a list of two points"""
    c = points[1] - points[0]
    c_normal = c / np.linalg.norm(c)
    mid_point = points[0] + c / 2
    new_points0 = mid_point + c / 2 + dist * c_normal
    new_points1 = mid_point - c / 2 - dist * c_normal
    return [new_points0, new_points1]


def collapse_list_2d(point_list):
    """reduce 3d vec to 2d vec"""
    point_extended = []
    for point in point_list:
        point_extended.append(point[0:2])
    return point_extended


def extend_list_3d(point_list, element=0):
    """extend 3d list to 4d list"""
    point_extended = []
    for point in point_list:
        point_extended.append(np.append(point, [element]))
    return point_extende


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


def cross_operator(v):
    """"Get 3d cross product operator such that v x u = V*u """
    V = np.zeros((3,3))
    V[0,:] = [0, -v[2], v[1]]
    V[1,:] = [v[2], 0, -v[0]]
    V[2,:] = [-v[1], v[0], 0]
    return V


def rotation_matrix(A,B):
    import tf
    au = A / np.linalg.norm(A)
    bu = B / np.linalg.norm(B)
    v = np.cross(au, bu)
    s = np.linalg.norm(v)
    if np.dot(au, bu)==1:
        R = np.identity(3)
    elif np.dot(au, bu)==-1:
        if au[0]==0 and au[2]==0:
            T = tf.transformations.rotation_matrix(np.pi, [1, 0, 0])
        elif au[1]==0 and au[0]==0:
            T = tf.transformations.rotation_matrix(np.pi, [1, 0, 0])
        elif au[2] == 0 and au[1]==0:
            T = tf.transformations.rotation_matrix(np.pi, [0,0,1])
        else:
            print ('[Rotation Matrix] Error weird symmetry attempted')
        R = T[0:3,0:3]
    else:
        c = np.dot(au,bu)
        Vx = cross_operator(v)
        R = np.identity(3) + Vx + np.matmul(Vx, Vx) * (1 - c) / s**2
    T = np.zeros((4,4))
    T[0:3,0:3] = R
    T[3,:] = T[:,3] =np.array([0,0,0,1])
    return T


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    new_dict = {}
    for key in x.keys():
        new_dict[key] = x[key] + y[key]
    return new_dict


def intersection(lst1, lst2):
    """get common elements between lst1 and lst2"""
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


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

np.random.seed(0)
seed1 = np.random.random(1000000)
seed2 = np.random.random(1000000)
counter = 0
def point_on_triangle(vertices):
    """get a random from within the boundaries of a triangle"""
    global counter
    A = vertices[0]
    B = vertices[1]
    C = vertices[2]
    r1 = seed1[counter]#random.uniform(0,1)
    r2 = seed2[counter]#random.uniform(0,1)
    counter += 1
    if counter == seed1.shape[0] or counter == seed2.shape[0]:
        counter = 0
    return (1 - np.sqrt(r1)) * A + (np.sqrt(r1) * (1 - (r2))) * B + (np.sqrt(r1) * r2) * C 
