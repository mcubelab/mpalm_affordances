import sys, os
sys.path.append(os.environ['CODE_BASE'] + '/catkin_ws/src/config/src')
from helper import helper, roshelper
import util
import sampling
import grasp_sampling, lever_sampling
from objects import Object, CollisionBody
import rospy, tf

#initialize ROS
rospy.init_node('planning_test')
listener = tf.TransformListener()
br = tf.TransformBroadcaster()

#1. build environment
# object_name = "cylinder_simplify.stl"
object_name = "realsense_box_experiments.stl"
gripper_name="mpalms_all_coarse.stl"
table_name="table_top_collision.stl"
_object = Object(mesh_name="config/descriptions/meshes/objects/" + object_name)
table = CollisionBody(mesh_name="config/descriptions/meshes/table/" + table_name)
trans, quat = listener.lookupTransform("yumi_body", "table_top", rospy.Time(0))
table_pose = trans + quat
table.setCollisionPose(table.collision_object, roshelper.list2pose_stamped(table_pose, "yumi_body"))
gripper_left = CollisionBody(mesh_name="config/descriptions/meshes/mpalm/" + gripper_name)
gripper_right = CollisionBody(mesh_name="config/descriptions/meshes/mpalm/" + gripper_name)
# q0 = roshelper.list2pose_stamped([0.4500000000000001, -0.040000000000000056, 0.07145000425107054, 0.4999999999999997, 0.4999999999999997, 0.4999999999999997, 0.5000000000000003])
q0 = roshelper.list2pose_stamped([0.45,0,0,0,0,0,1])

#3. sample object
sampler = sampling.Sampling(q0, _object, table, gripper_left, gripper_right, listener, br)
#4. grasp samplers
# grasp_samples = grasp_sampling.GraspSampling(sampler,
#                                              num_samples=3,
#                                              is_visualize=True)
#5. lever samples
lever_samples = lever_sampling.LeverSampling(sampler)
node_sequence, intersection_dict_lever = sampling.search_placement_graph(grasp_samples=None,
                                                                         lever_samples=lever_samples,
                                                                         placement_list=[1,3])

placement_sequence, sample_sequence, primitive_sequence = sampling.search_primitive_graph(_node_sequence=node_sequence,
                                                                                          intersection_dict_grasp=None,
                                                                                          intersection_dict_lever=intersection_dict_lever,
                                                                                          )

# node_sequence, intersection_dict_grasp, intersection_dict_lever = sampling.search_placement_graph(lever_samples,
#                                                                                                   grasp_samples,
from IPython import embed
embed()
print ('here')
# lever_samples = lever_sampling.LeverSampling(sampler)
# # grasp_samples.visualize()
# placement_list = deepcopy(self.placement_list)
# node_sequence, intersection_dict_grasp, intersection_dict_lever = sampling.search_placement_graph(lever_samples, grasp_samples, placement_list)
# placement_sequence, sample_sequence, primitive_sequence = sampling.search_primitive_graph(node_sequence, intersection_dict_grasp, intersection_dict_lever)
