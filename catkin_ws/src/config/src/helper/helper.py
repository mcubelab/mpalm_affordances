import numpy as np
import tf
import math

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

def cross_2d(u, v):
    """"Perform 2d cross product between u x v"""
    return u[0] * v[1] - u[1] * v[0]

def cross_operator(v):
    """"Get 3d cross product operator such that v x u = V*u """
    V = np.zeros((3,3))
    V[0,:] = [0, -v[2], v[1]]
    V[1,:] = [v[2], 0, -v[0]]
    V[2,:] = [-v[1], v[0], 0]
    return V

def rotation_matrix(A,B):
    """"Find the rotation matrix such that B=R*A"""
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

def collapse_list_2d(point_list):
    """reduce 3d vec to 2d vec"""
    point_extended = []
    for point in point_list:
        point_extended.append(point[0:2])
    return point_extended

def angle_2d(vec):
    """get heading of 2d vector"""
    angle = np.arctan2(vec[1], vec[0])
    return angle

def extend_list_3d(point_list, element=0):
    """extend 3d list to 4d list"""
    point_extended = []
    for point in point_list:
        point_extended.append(np.append(point, [element]))
    return point_extended

def vector_transform(T, r):
    """Perform 4x4 transformation matrix T to 3d vector r"""
    r_new = np.matmul(T, np.concatenate((r, np.array([1]))))
    return r_new[0:3]

def intersection(lst1, lst2):
    """get common elements between lst1 and lst2"""
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    new_dict = {}
    for key in x.keys():
        new_dict[key] = x[key] + y[key]
    return new_dict