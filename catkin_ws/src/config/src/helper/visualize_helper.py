import geometry_msgs
import numpy as np
import roshelper
from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import *

def visualize_final_object(qf, br, object_name):
    for i in range(4):
        roshelper.handle_block_pose(qf, br, 'yumi_body', 'object_final')
        visualize_object(qf,
                         filepath="package://config/descriptions/meshes/objects/" + object_name,
                         name = "/object_final",
                         color = (0., 0., 1., 1.),
                         scale = (0.99, 0.99, 0.99),
                         frame_id = "/yumi_body")
        rospy.sleep(.1)

def visualize_object_pose(q, object_name):
    for i in range(4):
        visualize_object(q,
                         filepath="package://config/descriptions/meshes/objects/" + object_name,
                         name = "/object",
                         color = (1.0, 126.0/255.0, 34.0/255.0, 1.),
                         frame_id = "/yumi_body")

def visualize_object(pose, filepath="package://config/descriptions/meshes/objects/object.stl",name = "/object", color = (0., 0., 1., 1.), frame_id = "/yumi_body", scale = (1., 1., 1.)):
    marker_pub = rospy.Publisher(name, Marker, queue_size=1)
    marker_type = Marker.MESH_RESOURCE
    marker = Marker()
    marker.header.frame_id  = frame_id
    marker.ns = name
    marker.header.stamp = rospy.Time(0)
    marker.action =  Marker.ADD
    marker.pose.orientation.w =  1.0
    marker.type = marker_type
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.lifetime.secs  = 1
    marker.pose = pose.pose
    marker.mesh_resource = filepath
    marker.lifetime = rospy.Duration(10000)

    for i in range(10):
        marker_pub.publish(marker)

def plot_meshes(mesh_list=[], point_list=[], points_new_list=[]):
    # Optionally render the rotated cube faces
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    # Render the cube
    color_list = ['r', 'b', 'g']
    for counter, mesh in enumerate(mesh_list):
        poly_3d = mplot3d.art3d.Poly3DCollection(mesh.vectors)
        axes.add_collection3d(poly_3d)
        poly_3d.set_facecolor(color_list[counter])

    for point in point_list:
        axes.scatter([point[0]],
                       [point[1]],
                       [point[2]],
                       c=[1,0,0],
                       alpha=0.5)

    for point in points_new_list:
        axes.scatter([point[0]],
                       [point[1]],
                       [point[2]],
                       c=[0,1,0],
                       alpha=1)

    # Auto scale to the mesh size
    if len(mesh_list)>0:
        scale = mesh.points.flatten(-1)
        axes.auto_scale_xyz(scale, scale, scale)
    plt.xlabel('x')
    plt.ylabel('y')
    # Show the plot to the screen
    plt.show()


def delete_markers(name = "/proposals", ns = ''):
    marker_pub = rospy.Publisher(name, Marker, queue_size=1)
    marker = Marker()
    marker.action = 3  # delete all
    marker.ns = ns
    for i in range(10):
        marker_pub.publish(marker)
        rospy.sleep(.05)


def visualize_grasps(br, grasp_list, pose, name = "/proposals", color = (1., 0., 0., 1.), id=0):
    if len(grasp_list)==2:
        grasp_list = [grasp_list]
    for i in range(2):
        roshelper.handle_block_pose(pose, br, "/yumi_body", "/proposals_base")

    marker_pub = rospy.Publisher(name, Marker, queue_size=1)
    marker_type = Marker.LINE_LIST

    marker = Marker()
    marker.header.frame_id  = "/proposals_base"
    marker.ns = name
    marker.header.stamp = rospy.Time(0)
    marker.action = Marker.ADD
    marker.type = marker_type
    marker.scale.x = .001
    marker.scale.y = .001
    marker.scale.z = .001
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.id = id

    marker.lifetime = rospy.Duration(10000)

    for points in grasp_list:
        points_elon = elongate_vector(points, 0.03)
        for point in points_elon:
            p =geometry_msgs.msg.Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            marker.points.append(p)

    for i in range(5):
        marker_pub.publish(marker)
        rospy.sleep(.05)

def processFeedback(feedback):
    msg = geometry_msgs.msg.PoseStamped()
    msg.header = feedback.header
    msg.header.frame_id = "yumi_body"
    msg.pose = feedback.pose
    msg.pose.position.z -= 0.1
    pub = rospy.Publisher(feedback.client_id, geometry_msgs.msg.PoseStamped, latch=True)
    pub.publish(msg)

def createMeshMarker(resource, offset=(0, 0, 0), rgba=(1, 0, 0, 1), orientation=(0, 0, 0, 1), scale=1, scales=(1, 1, 1),
                     frame_id="/world"):

    marker = Marker()
    marker.mesh_resource = resource;
    marker.header.frame_id = frame_id
    marker.type = marker.MESH_RESOURCE
    marker.scale.x = scale * scales[0]
    marker.scale.y = scale * scales[1]
    marker.scale.z = scale * scales[2]
    marker.color.a = rgba[3]
    marker.color.r = rgba[0]
    marker.color.g = rgba[1]
    marker.color.b = rgba[2]
    marker.pose.orientation.x = orientation[0]
    marker.pose.orientation.y = orientation[1]
    marker.pose.orientation.z = orientation[2]
    marker.pose.orientation.w = orientation[3]
    marker.pose.position.x = offset[0]
    marker.pose.position.y = offset[1]
    marker.pose.position.z = offset[2]

    obj_control = InteractiveMarkerControl()
    obj_control.always_visible = True
    obj_control.markers.append(marker)

    return obj_control


def createMoveControls(fixed=False):
    controls = []
    ## rotate control x
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 1
    control.orientation.y = 0
    control.orientation.z = 0
    control.name = "rotate_x"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    if fixed:
        control.orientation_mode = InteractiveMarkerControl.FIXED
    controls.append(control)

    ## rotate control y
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 1
    control.orientation.z = 0
    control.name = "rotate_y"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    if fixed:
        control.orientation_mode = InteractiveMarkerControl.FIXED
    controls.append(control)

    ## rotate control z
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 0
    control.orientation.z = 1
    control.name = "rotate_z"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    if fixed:
        control.orientation_mode = InteractiveMarkerControl.FIXED
    controls.append(control)

    ## move control x
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 1
    control.orientation.y = 0
    control.orientation.z = 0
    control.name = "move_x"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    if fixed:
        control.orientation_mode = InteractiveMarkerControl.FIXED
    controls.append(control)

    ## move control y
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 1
    control.orientation.z = 0
    control.name = "move_y"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    if fixed:
        control.orientation_mode = InteractiveMarkerControl.FIXED
    controls.append(control)

    ## move control z
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 0
    control.orientation.z = 1
    control.name = "move_z"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    if fixed:
        control.orientation_mode = InteractiveMarkerControl.FIXED
    controls.append(control)

    return controls

def createInteractiveMarker(pose, scale=0.3, frame_id="world", name='my_marker', description="Marker description"):
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = frame_id
    int_marker.name = name
    int_marker.description = description
    int_marker.scale = 0.1
    int_marker.pose = pose.pose
    return int_marker

def elongate_vector(points, dist):
    """create a line of dist from a list of two points"""
    c = points[1] - points[0]
    c_normal = c / np.linalg.norm(c)
    mid_point = points[0] + c / 2
    new_points0 = mid_point + c / 2 + dist * c_normal
    new_points1 = mid_point - c / 2 - dist * c_normal
    return [new_points0, new_points1]
