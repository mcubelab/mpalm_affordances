import sys, os
sys.path.append(os.environ['CODE_BASE'] + '/catkin_ws/src/config/src')
from helper import helper, roshelper
import util
import sampling
import grasp_sampling, lever_sampling
from objects import Object, CollisionBody
import numpy as np
import rospy, tf
from IPython import embed
import time

#initialize ROS
rospy.init_node('planning_test')
listener = tf.TransformListener()
br = tf.TransformBroadcaster()

#1. build environment
# object_name = "cylinder_simplify.stl"
# object_name = "realsense_box_experiments.stl"
# object_name = "mustard_coarse_inertia_trans.stl"
# object_name = "mustard_coarse.stl"
# object_name = 'mustard_coarse_2.stl'
# object_name = 'mustard_centered.stl'
object_name = 'mustard_1k.stl'

# object_name = 'test_cuboid_smaller_4206.stl'  # block where things work
# object_name = 'test_cuboid_smaller_3206.stl'  # block where things do not work  

# object_name = 'test_cuboid_smaller_%d.stl' % 1025
print("Object Name: " + object_name) 
gripper_name="mpalms_all_coarse.stl"
table_name="table_top_collision.stl"

meshes_dir = '/root/catkin_ws/src/config/descriptions/meshes'
# object_mesh = os.path.join(meshes_dir, 'objects/cuboids/', object_name)
object_mesh = os.path.join(meshes_dir, 'objects/', object_name)
table_mesh = os.path.join(meshes_dir, 'table', table_name)
gripper_mesh = os.path.join(meshes_dir, 'mpalm', gripper_name)

# _object = Object(mesh_name="config/descriptions/meshes/objects/" + object_name)
# table = CollisionBody(mesh_name="config/descriptions/meshes/table/" + table_name)
start_time = time.time()
_object = {}
_object['file'] = object_name
_object['object'] = Object(mesh_name=object_mesh)
table = CollisionBody(mesh_name=table_mesh)
trans, quat = listener.lookupTransform("yumi_body", "table_top", rospy.Time(0))
table_pose = trans + quat
table.setCollisionPose(table.collision_object, roshelper.list2pose_stamped(table_pose, "yumi_body"))
# gripper_left = CollisionBody(mesh_name="config/descriptions/meshes/mpalm/" + gripper_name)
# gripper_right = CollisionBody(mesh_name="config/descriptions/meshes/mpalm/" + gripper_name)
gripper_left = CollisionBody(mesh_name=gripper_mesh)
gripper_right = CollisionBody(mesh_name=gripper_mesh)

# q0 = roshelper.list2pose_stamped([0.4500000000000001, -0.040000000000000056, 0.07145000425107054, 0.4999999999999997, 0.4999999999999997, 0.4999999999999997, 0.5000000000000003])
q0 = roshelper.list2pose_stamped([0.45,0,0,0,0,0,1])

#3. sample object
sampler = sampling.Sampling(q0, _object, table, gripper_left, gripper_right, listener, br)
#4. grasp samplers
print("getting grasp samples")
grasp_samples = grasp_sampling.GraspSampling(sampler,
                                             num_samples=1,
                                             is_visualize=True) 
print('total grasp sampling time: ' + str(time.time() - start_time))

inter_dict, graph = sampling.build_placement_graph(grasp_samples=grasp_samples, placement_list=None)
placement_list = [1, 5]
graph_layer_1 = sampling.add_boundary_edges(placement_list, graph)
placement_sequence = graph_layer_1.dijkstra('start', 'end')
print(placement_sequence)
embed()

try:
    node_sequence, intersection_dict_grasp = sampling.search_placement_graph(grasp_samples=grasp_samples,
                                                                            placement_list=placement_list)
except KeyError as e:
    print(e)
    embed()                                                                        

placement_sequence, sample_sequence, primitive_sequence = sampling.search_primitive_graph(_node_sequence=node_sequence,
                                                                                          intersection_dict_grasp=intersection_dict_grasp)
print('total time: ' + str(time.time() - start_time))
#5. lever samples
# lever_samples = lever_sampling.LeverSampling(sampler)
# node_sequence, intersection_dict_lever = sampling.search_placement_graph(grasp_samples=None,
#                                                                          lever_samples=lever_samples,
#                                                                          placement_list=[1,3])

# placement_sequence, sample_sequence, primitive_sequence = sampling.search_primitive_graph(_node_sequence=node_sequence,
#                                                                                           intersection_dict_grasp=None,
#                                                                                           intersection_dict_lever=intersection_dict_lever,
#                                                                                           )

# node_sequence, intersection_dict_grasp, intersection_dict_lever = sampling.search_placement_graph(lever_samples,
#                                                                                                   grasp_samples,

# import trimesh
# new_fname = '/root/catkin_ws/src/config/descriptions/meshes/objects/mustard_coarse.stl'
# tmesh = trimesh.load_mesh(new_fname)
embed()
print ('here')

# lever_samples = lever_sampling.LeverSampling(sampler)
grasp_samples.visualize(object_mesh_file=object_name)

embed()
# placement_list = deepcopy(self.placement_list)
# node_sequence, intersection_dict_grasp, intersection_dict_lever = sampling.search_placement_graph(lever_samples, grasp_samples, placement_list)
# placement_sequence, sample_sequence, primitive_sequence = sampling.search_primitive_graph(node_sequence, intersection_dict_grasp, intersection_dict_lever)
