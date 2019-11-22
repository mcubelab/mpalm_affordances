from copy import deepcopy
import csv
import numpy as np
import os

g = 9.81
np.random.seed(0)
seed1 = np.random.random(1000000)
seed2 = np.random.random(1000000)
counter = 0

def get_elements_from_dictionary(data, index_start, index_end):
    new_data = {}
    for key in data.keys():
        new_data[key] = []
        if type(data[key])==list:
            for item in data[key][index_start:index_end]:
                new_data[key].append(item)
        else:
            new_data[key] = data[key]
    return new_data

def concatenate_elements_dictionary(data1, data2):
    new_data = {}
    for key in data1.keys():
        new_data[key] = []
        if type(data1[key])==list:
            if key in data1:
                for item1 in data1[key]:
                    new_data[key].append(item1)
            if key in data2:
                for item2 in data2[key]:
                    new_data[key].append(item2)
        else:
            new_data[key] = data1[key]
    return new_data

def merge_dictionaries(data1, data2):
    new_dict = {}
    for key in data1:
        new_dict[key] = data1[key]
    for key2 in data2:
        new_dict[key2] = data2[key2]
    return new_dict

def terminate_ros_node(s):
    import subprocess
    list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
    list_output = list_cmd.stdout.read()
    retcode = list_cmd.wait()
    assert retcode == 0, "List command returned %d" % retcode
    for term in list_output.split("\n"):
        if (term.startswith(s)):
            os.system("rosnode kill " + term)
            print ("rosnode kill " + term)

def csv2dict(filename):
    # input_file = csv.DictReader(open(filename))
    with open (filename) as f:
        data = f.read()

    reader = csv.DictReader(data.splitlines(0)[0:])
    lines = list(reader)
    # for row in reader:
    # counties = {k: v for (k,v in ((line['%time']) for line in lines)}
    name_list = lines[0].keys()
    d = {}
    for key in name_list:
        d[key] = []
        for data in range(len(lines)):
            d[key].append(float(lines[data][key]))
    return d

def initialize_rosbag(topics, exp_name='test'):
    import datetime
    import subprocess
    #Saving rosbag options
    name_of_bag  = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + '_' + exp_name
    # topics = ["/viewer/image_raw"]
    # topics = ["/time", "/xc", "/uc", "/us", "/q_pusher_sensed", "/q_pusher_commanded", "/viewer/image_raw", "/viewer/image_raw/compressed", "/viewer/camera_info", "/frame/track_start_frame_pub", "/frame/vicon_world_frame_pub", "/frame/viewer_frame_pub"]
    #topics = ["/viewer/image_raw"]
    #topics = []
    dir_save_bagfile = os.environ['CODE_BASE'] + '/data/rosbag_data/'
    rosbag_proc = subprocess.Popen('rosbag record -q -O %s %s' % (name_of_bag, " ".join(topics)) , shell=True, cwd=dir_save_bagfile)

def terminate_rosbag():
    terminate_ros_node('/record')

def reorder_joints_list_trajectory(joints_list):
    joints_array = np.array(joints_list)
    joints_left = list(joints_array[:, 0, :])
    joints_right = list(joints_array[:, 1, :])
    joints_left = deorder_joints(joints_left)
    joints_right = deorder_joints(joints_right)
    joint_traj = [joints_left, joints_right]
    return joint_traj

def velocities_from_joints(joints_list, dt):
    velocities_joints_list = []
    for n in range(len(joints_list)-1):
        # 1. compute joints distance vector
        joints_n = joints_list[n]
        joints_next = joints_list[n+1]
        delta_joints = np.array(joints_next) - np.array(joints_n)
        # 2. compute velocity of each joint (in rad/s)
        velocity_joints = delta_joints / dt
        velocities_joints_list.append(velocity_joints)
    return  velocities_joints_list

def reorder_joint_list(joint_list, single=False):
    joints_left = deorder_joints(joint_list[0])
    joints_right = deorder_joints(joint_list[1])
    if single:
        traj = [joints_left, joints_right]
    else:
        traj = [[joints_left, joints_left],
                [joints_right, joints_right]]
    return traj


def simplify_axis(ax, xbottom=True, xtop=False,  yleft=True, yright=False):
    import matplotlib.pyplot as plt
    """
    Remove white spacing from axis. Make lines around axis bold.
    @param ax The axis to simplify
    @param xbottom If true, draw the bottom axis - otherwise remove the line
    @param xtop If true, draw the top axis - otherwise remove the lin
    @param yleft If true, draw the left axis - otherwise remove the line
    @param yright If true, draw the right axis - otherwise remove the line
    """

    ax.set_frame_on(False)
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()

    if yleft:
        ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax),
                                   color='black', linewidth=1,
                                   zorder=100, clip_on=False))
        ax.get_yaxis().tick_left()

    if yright:
        ax.add_artist(plt.Line2D((xmax, xmax), (ymin, ymax),
                                   color='black', linewidth=1,
                                   zorder=100, clip_on=False))
        ax.get_yaxis().tick_right()

    if yleft and yright:
        ax.get_yaxis().set_ticks_position('both')

    if xbottom:
        ax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin),
                                   color='black', linewidth=1,
                                   zorder=100, clip_on=False))
        ax.get_xaxis().tick_bottom()

    if xtop:
        ax.add_artist(plt.Line2D((xmin, xmax), (ymax, ymax),
                                   color='black', linewidth=1,
                                   zorder=100, clip_on=False))
        ax.get_xaxis().tick_top()

    if xbottom and xtop:
        ax.get_xaxis().set_ticks_position('both')

def configure_fonts(fontsize=35, legend_fontsize=8,
                      usetex=True, figure_dpi=300):
    import matplotlib.pyplot as plt

    """
    Configure fonts. Fonts are set to serif .
    @param fontsize The size of the fonts used on the axis and within the figure
    @param legend_fontsize The size of the legend fonts
    @param usetex If true, configure to use latex for formatting all text
    @param figure_dpi Set the dots per inch for any saved version of the plot
    """
    plt.rcParams.update({
            'font.family':'serif',
            'font.serif':'Computer Modern Roman',
            'font.size': fontsize,
            'legend.fontsize': legend_fontsize,
            'legend.labelspacing': 0,
            'text.usetex': usetex,
            'savefig.dpi': figure_dpi
    })


def output(fig, path, size, fontsize=8, legend_fontsize=8, latex=True):
    """
    Save the figure.
    @param fig The figure to save
    @param path The output path to write the figure to
       (if None, figure not saved, just rendered to screen)
    @param size The size of the output figure
    @param fontsize The size of the font
    @param legend_fontsize The size of the legend font
    @param latex If true, use latex to render the text in the figure
    """
    if path is not None:
        configure_fonts(fontsize=fontsize, legend_fontsize=legend_fontsize,
                        usetex=latex)
        fig.set_size_inches(size)
        fig.savefig(path, pad_inches=0.02, bbox_inches='tight')
    else:
        plt.show()


def extend_vec(vec, index, is_time=False, middle_factor=3):
    if is_time:
        dif_time = vec[1] - vec[0]
    index_middle = len(vec) / middle_factor
    value_middle = vec[index_middle]
    vec = list(vec)
    for i in range(index):
        if is_time:
            value_middle += dif_time
            vec.insert(index_middle, value_middle)
            vec[index_middle:] = np.array(vec[index_middle:]) + dif_time
        else:
            vec.insert(index_middle, value_middle)
    return vec

def load_data(filename):
    import pickle
    # filename = os.environ['CODE_BASE'] + '/data/' + filename
    try:
        with open(filename) as f:
            data = pickle.load(f)
            return data
    except NameError:
        print ("Could not load data: ", NameError)

def save_data(data, filename):
    import pickle
    # filename = os.environ['CODE_BASE'] + '/data/' + filename
    try:
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    except NameError:
        print ("Could not save data: ", NameError)


def load_csv(filename, experiment_name, key_list):
    # 0. hyperparemeter
    dict_data = {}
    for key in key_list:
        print ('key: ', key)
        dict_data[key] = {}
        dict_data[key]['name'] = None
        dict_data[key]['value'] = []
        #'/home/fishbowl/mpalms/data/rosbag_data/' + filename + 'csv/' + filename + '__' + key + '.csv'
        csv_filename = filename + experiment_name + '__' + key + '.csv'
        with open(csv_filename) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for counter, row in enumerate(readCSV):
                if counter == 0:
                    dict_data[key]['name'] = row
                else:
                    dict_data[key]['value'].append(row)
    return dict_data

def find_union_arrays(A, B):
    A = A.astype(float)
    B = B.astype(float)
    if A.shape[0]<B.shape[0]:
        diff_shape = B.shape[0] - A.shape[0]
        A = np.concatenate((A, 1000.0 * np.ones((diff_shape,3))))
    elif B.shape[0]<A.shape[0]:
        diff_shape = A.shape[0] - B.shape[0]
        B = np.concatenate((B, 1000.0 * np.ones((diff_shape,3))))

    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [A.dtype]}

    C = np.intersect1d(A.view(dtype), B.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C

def is_arr_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if elem is myarr), False)

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def angle_2d(vec):
    angle = np.arctan2(vec[1], vec[0])
    return angle

def height(vec):
    return vec[2]

def cross_2d(u, v):
    return u[0] * v[1] - u[1] * v[0]

def cross_operator(v):
    V = np.zeros((3,3))
    V[0,:] = [0, -v[2], v[1]]
    V[1,:] = [v[2], 0, -v[0]]
    V[2,:] = [-v[1], v[0], 0]
    return V

def deorder_joints(joints):
    if len(joints)==7:
        return [joints[x] for x in [0,1,3,4,5,6,2]]
    else:
        new_list = []
        for term in joints:
            new_list.append([term[x] for x in [0,1,3,4,5,6,2]])
        return new_list

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


def point2line_distance(_p, _q, _rs):
    p = np.array(_p)
    q = np.array(_q)
    rs = np.array(_rs)
    x = p-q
    return np.linalg.norm(
        np.outer(np.dot(rs-q, x)/np.dot(x, x), x)+q-rs,
        axis=1)[0]

def point_on_triangle(vertices):
    global counter
    A = vertices[0]
    B = vertices[1]
    C = vertices[2]
    r1 = seed1[counter]#random.uniform(0,1)
    r2 = seed2[counter]#random.uniform(0,1)
    counter += 1
    return (1 - np.sqrt(r1)) * A + (np.sqrt(r1) * (1 - (r2))) * B + (np.sqrt(r1) * r2) * C

def project_point2plane(point, plane_normal, plane_points):
    point_plane = plane_points[0]
    w = point - point_plane
    dist = (np.dot(plane_normal, w) / np.linalg.norm(plane_normal))
    projected_point = point - dist * plane_normal / np.linalg.norm(plane_normal)
    return projected_point, dist

def point_in_triangle(query_point,
                      vertices):
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

def sort_length(val_dict):
    return val_dict['length']

def extend_list_3d(point_list, element=0):
    point_extended = []
    for point in point_list:
        point_extended.append(np.append(point, [element]))
    return point_extended

def extend_point_3d(point):
    return np.array([point[0], point[1], 1])

def collapse_list_2d(point_list):
    point_extended = []
    for point in point_list:
        point_extended.append(point[0:2])
    return point_extended


def sort_angle(val_dict):
    return val_dict['contact_angle']

def sort_N_star(val_dict):
    return val_dict['N_star']

def find_mid_point(point_list):
    x_list, y_list, z_list = extract_variables(point_list)
    x = np.mean(x_list)
    y = np.mean(y_list)
    z = np.mean(z_list)
    return [x,y,z]

def find_highest_points(point_list, n=2):
    point_list.sort(key=height, reverse=True)
    return point_list[0:2]

def extract_variables(list_points, three_d=True):
    x_list = []
    y_list = []
    if three_d:
        z_list = []
    for points in list_points:
        x_list.append(points[0])
        y_list.append(points[1])
        if three_d:
            z_list.append(points[2])
    if three_d:
        return x_list, y_list, z_list
    else:
        return x_list, y_list

def plot_list_points(list_points):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_list, y_list, z_list = extract_variables(list_points)
    ax.scatter(x_list, y_list, z_list, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

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

def transform_point_list(face, T):
    point_list = []
    for i in range(len(face)):
        rotated_face = np.matmul(T, np.append(face[i], 1))
        point_list.append(rotated_face[0:3])
    return point_list

def matrix2vec(T):
    x_vec = T[0:3,0]
    y_vec = T[0:3,1]
    z_vec = T[0:3,2]
    return x_vec, y_vec, z_vec

def merge_two_dicts(x, y):
    new_dict = {}
    for key in x.keys():
        new_dict[key] = x[key] + y[key]
    """Given two dicts, merge them into a new dict as a shallow copy."""
    return new_dict

def axis_from_points(list_points, vec_guide=None):
    assert(len(list_points)==2, "length of list of points must be 2")
    delta_points = np.array(list_points[1]) - np.array(list_points[0])
    delta_points = delta_points / np.linalg.norm(delta_points)
    if vec_guide is not None:
        if np.dot(delta_points, vec_guide) < 0:
            delta_points = -delta_points
    return delta_points

def initialize_system_poses(q0_planar=np.array([0.3, 0.0, 0 * np.pi / 180]), qf_planar=np.array([0.3, -0.1, 30 * np.pi / 180]), placement_list=[0, 5], object=None):
    import roshelper
    q0 = roshelper.get3dpose_object(q0_planar,
                                    object,
                                    stable_placement=placement_list[0])
    qf = roshelper.get3dpose_object(qf_planar,
                                    object,
                                    stable_placement=placement_list[-1])
    sampling_base_pose = roshelper.get3dpose_object(q0_planar,
                                                    object,
                                                    stable_placement=0)  # note: sampling frame must be with stable_placement=0
    return q0, qf, sampling_base_pose

def vector_from_points(point_list, is_normalize=False):
    if is_normalize:
        return (np.array(point_list[1]) - np.array(point_list[0])) / \
               np.linalg.norm((np.array(point_list[1]) - np.array(point_list[0])))
    else:
        return np.array(point_list[1]) - np.array(point_list[0])

def get_initial_states_rviz(object=None):
    from geometry_msgs.msg import PoseStamped
    import rospy
    import roshelper
    q0 = rospy.wait_for_message('/rviz/q0_pose', PoseStamped)
    qf = rospy.wait_for_message('/rviz/qf_pose', PoseStamped)

    q0_planar, placement_0 = roshelper.get2dpose_object(q0, object)
    qf_planar, placement_f = roshelper.get2dpose_object(qf, object)
    sampling_base_pose = roshelper.get3dpose_object(q0_planar,
                                                    object,
                                                    stable_placement=0)
    return q0, qf, sampling_base_pose, [placement_0, placement_f]

def jacobian_2d(px, py):
    return np.array([[1,0,-py],
            [0,1,px]])

def C3_2d(theta, is_symbolic=False):
    if is_symbolic:
        from sympy import sin, cos
        C = np.array([[cos(theta), sin(theta)],
                      [-sin(theta), cos(theta)]]
                     )
    else:
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

def C_tilde_fun(theta, is_symbolic=False):
    if is_symbolic:
        from sympy import sin, cos
        C = np.array([[cos(theta), sin(theta), 0],
                      [-sin(theta), cos(theta), 0],
                      [0, 0, 1]])
    else:
        C = np.array([[np.cos(theta), np.sin(theta), 0],
                      [-np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]]
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

def rotate_2d_pose_origin(pose, angle, origin, Flip=False):
    r = pose[0:2] - origin
    theta = pose[2]
    R = C3_2d(angle).transpose()
    r_new = origin + np.matmul(R, r)
    if Flip:
        theta_new = -theta
    else:
        theta_new = theta
    pose_new = np.array([r_new[0], r_new[1], theta_new])
    return pose_new

def rotate_2d_pose(pose, angle, origin):
    r = pose[0:2] - origin
    theta = pose[2]
    R = C3_2d(angle).transpose()
    r_new = origin + np.matmul(R, r)
    theta_new = theta + angle
    pose_new = np.array([r_new[0], r_new[1], theta_new])
    return pose_new

def vector_transform(T, r):
    r_new = np.matmul(T, np.concatenate((r, np.array([1]))))
    return r_new[0:3]

def multimul(matrix_list):
    mat = matrix_list[0]
    for i in range(1, len(matrix_list)):
        mat = np.matmul(mat, matrix_list[i])
    return mat

def transform_list_points(T, list_points):
    rotated_points = []
    for point in list_points:
        rotated_points.append(vector_transform(T, point))
    return rotated_points

def abb2ros_pose(pose_abb):
    print (pose_abb)
    pose_new = np.zeros(7)
    pose_new[0:3] = np.array(pose_abb[0:3]) / 1000
    pose_new[3] = pose_abb[4]
    pose_new[4] = pose_abb[5]
    pose_new[5] = pose_abb[6]
    pose_new[6] = pose_abb[3]
    return pose_new

def find_interpolate_frac(t, t_star_vec):
    closest_index = np.argmin(np.abs(t - np.array(t_star_vec)))
    closest_value = t_star_vec[closest_index]
    # find closest value (below)

    if t - closest_value < 0:  # closest value is above t
        high_index = deepcopy(closest_index)
        for i in range(100):
            closest_index -= 1
            if t - t_star_vec[closest_index] > 0:
                if closest_index == -1:
                    low_index = 0
                    break
                else:
                    low_index = closest_index
                    break
    else:  # closest value is below t
        low_index = deepcopy(closest_index)
        for i in range(100):
            closest_index += 1
            if closest_index < len(t_star_vec) - 1:
                if t - t_star_vec[closest_index] < 0:
                    high_index = closest_index
                    break
            else:
                high_index = closest_index - 1
                break
    if (t_star_vec[high_index] - t_star_vec[low_index])==0:
        frac=0
    else:
        frac = (t - t_star_vec[low_index]) / (t_star_vec[high_index] - t_star_vec[low_index])
    return frac, low_index, high_index

def interpolate_pose(frac, low_index, high_index, pose_list):
    import roshelper
    if high_index>len(pose_list)-1:
        high_index = len(pose_list) - 1
    if low_index > len(pose_list) - 1:
        low_index = len(pose_list) - 1
    if high_index < 0:
        high_index = 0
    if low_index < 0:
        low_index = 0

    pose_interp = roshelper.interpolate_pose_discrete(pose_list[low_index],
                                                           pose_list[high_index],
                                                           frac)
    return pose_interp

def interpolate_dual_joints(frac, low_index, high_index, joints_list):
    joints_array = np.array(joints_list)
    joints_left = joints_array[:,0,:]
    joints_right = joints_array[:, 1, :]
    if high_index>len(joints_list)-1:
        high_index = len(joints_list) - 1
    if low_index > len(joints_list) - 1:
        low_index = len(joints_list) - 1
    if high_index < 0:
        high_index = 0
    if low_index < 0:
        low_index = 0

    joints_interp_left = interpolate_joints(frac, low_index, high_index, joints_left)
    joints_interp_right = interpolate_joints(frac, low_index, high_index, joints_right)
    return [joints_interp_left, joints_interp_right]

def interpolate_joints(frac, low_index, high_index, joints_array):
    joints_interp = []
    for i in range(joints_array.shape[1]):
        val = np.interp(frac, [0, 1], [joints_array[low_index][i], joints_array[high_index][i]])
        joints_interp.append(val)
    return joints_interp

def linspace_array(start, final, N):
    #1. convert list to array
    start_array = np.array(start)
    final_array = np.array(final)

    #2. interpolate all elements individually
    interp_list = []
    for element in range(len(start_array)):
        interp_list.append(np.linspace(start_array[element], final_array[element], N))
    #3. reshape output to match input shape
    interp_array = np.swapaxes(np.array(interp_list), 0, 1)
    if type(start_array)==list:
        return list(interp_array)
    else:
        return interp_array

def yumi2robot_joints(joints, deg=True):
    reorder_list = [0,1,6,2,3,4,5]
    joints_new = [0] * 7
    for i in range(len(joints)):
        if deg:
            joints_new[i] = joints[reorder_list[i]] * (np.pi / 180)
        else:
            joints_new[i] = joints[reorder_list[i]]
    return joints_new

def robot2yumi_joints(joints, deg=False):
    reorder_list = [0,1,3,4,5,6,2]
    joints_new = [0] * 7
    for i in range(len(joints)):
        if deg:
            joints_new[i] = joints[reorder_list[i]]
        else:
            joints_new[i] = joints[reorder_list[i]] * (180 / np.pi)
    return joints_new

def velocity_from_poses(pose0, pose1, dt):
    # pose1 and pose2 are 7-dimensional vectors (x, y, z)
    # vel is a 6-dimensional vector (qx, qy, qz, qw)
    vel = []
    vel.append((pose1.position.x-pose0.position.x)/dt)
    vel.append((pose1.position.y-pose0.position.y)/dt)
    vel.append((pose1.position.z-pose0.position.z)/dt)
    q0 = np.quaternion(pose0.orientation.w, pose0.orientation.x, pose0.orientation.y, pose0.orientation.z)
    q1 = np.quaternion(pose1.orientation.w, pose1.orientation.x, pose1.orientation.y, pose1.orientation.z)
    q = q1/q0
    len = np.sqrt(q.x * q.x + q.y * q.y + q.z * q.z)
    angle = 2*np.arctan2(len, q.w);
    if len > 0:
        axis = [q.x/len, q.y/len, q.z/len]
    else:
        axis = [1.0, 0.0, 0.0]
    vel.append(axis[0]*angle/dt)
    vel.append(axis[1]*angle/dt)
    vel.append(axis[2]*angle/dt)
    return vel

def get_angle_correction(direction_world, axis):
    axis_angle = np.arctan2(axis[1], axis[0])
    direction_angle = np.arctan2(direction_world[1], direction_world[0])
    dif_angle = axis_angle - direction_angle
    return dif_angle

def save_python_object(data, filename):
    import os
    try:
        os.mkdir(filename)
    except:
        pass
    save_data(data, filename)
