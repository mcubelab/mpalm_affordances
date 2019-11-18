import sys, os
sys.path.append(os.environ['CODE_BASE'] + '/catkin_ws/src/config/src')
import itertools
import numpy as np
import rospy
from helper import helper
from helper import roshelper
from helper import visualize_helper
import util
from copy import deepcopy
import tf.transformations as tfm
from geometry_msgs.msg import PoseStamped
import objects

class GraspSampling(object):
    def __init__(self, sampler, num_samples, is_visualize=False):
        self.sampler = sampler
        self.samples_dict = {}
        self.samples_dict['placement_id'] = []
        self.samples_dict['sample_ids'] = []
        self.samples_dict['points'] = []
        self.samples_dict['face_normals'] = []
        self.samples_dict['gripper_poses'] = []
        self.samples_dict['grasp_width'] = []
        self.samples_dict['base_pose'] = []
        self.samples_dict['object_pose'] = []
        self.samples_dict['T'] = []
        self.samples_dict['T_pose'] = []
        self.samples_dict['collisions'] = []
        #1. generate contact point samples in nominal reference frame
        self.generate_grasp_samples(num_samples)
        #2. from contact points, generate grasp samples in all stable placements
        self.initialize_stable_placements()
        #3. filter out samples that collide with table
        self.remove_collisions()
        #4. visualize samples
        if is_visualize:
            self.visualize(flag="placements")

    def generate_grasp_samples(self, num_samples = 100, angle_threshold =5 * np.pi / 180):
        """sample faces of object mesh for bipolar grasps"""
        #1. extract properties from stl mesh
        faces = self.sampler.object.faces
        normals = self.sampler.object.normals
        faces_index = range(len(faces))
        grasp_index_list = []
        grasp_combination_index_list = list(itertools.combinations(faces_index, 2))
        #2. find and filter potential face pairs for grasping
        for face_pair in grasp_combination_index_list:
            u = -normals[face_pair[0]]
            v = normals[face_pair[1]]
            cos_angle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            angle = abs(np.arccos(cos_angle))
            if angle < angle_threshold and face_pair[0] is not face_pair[1]:
                grasp_index_list.append(face_pair)
            elif abs(angle - np.pi) < angle_threshold and face_pair[0] is not face_pair[1]:
                grasp_index_list.append(face_pair)

        #3. sample grasps (about default orientation in free space) and add to dict_grasp
        self.grasp_proposal_list = []
        self.grasp_index_list = grasp_index_list
        self.dict_grasp = {}
        self.dict_grasp['points'] = []
        self.dict_grasp['face_pair'] = []
        self.dict_grasp['index'] = []
        self.dict_grasp['angle'] = []
        self.dict_grasp['face_normals'] = []
        self.dict_grasp['sample_ids'] = []
        counter_sample = 0
        for counter, face_pair in enumerate(grasp_index_list):
            for sample in range(0, num_samples):
                # sample on first face
                point = util.point_on_triangle(faces[face_pair[0]])
                # project to other plan
                projected_point, dist = util.project_point2plane(point, normals[face_pair[1]], faces[face_pair[1]])
                #check if within triangle
                in_triangle = util.point_in_triangle(projected_point, faces[face_pair[1]])
                is_far_enough = self.point_far_from_samples(point)
                if in_triangle and is_far_enough:
                    self.dict_grasp['points'].append([point, projected_point])
                    self.dict_grasp['angle'].append(0)
                    self.dict_grasp['sample_ids'].append(counter_sample)
                    self.dict_grasp['face_pair'].append(face_pair)
                    self.dict_grasp['index'].append(counter)
                    self.dict_grasp['face_normals'].append([normals[face_pair[0]], normals[face_pair[1]]])
                    counter_sample += 1

    def visualize(self, flag="collisions"):
        if flag=="placements" or flag=="all" or flag=="collisions":
            for i in range(len(self.samples_dict['object_pose'])):
                visualize_helper.delete_markers()
                roshelper.handle_block_pose(self.samples_dict['object_pose'][i][0],
                                            self.sampler.br,
                                            'proposals_base',
                                            'object')

                roshelper.handle_block_pose(self.sampler.base_initial,
                                            self.sampler.br,
                                            'yumi_body',
                                            'proposals_base')

                visualize_helper.visualize_grasps(self.sampler.br,
                                                  self.samples_dict['points'][i],
                                                  pose=self.samples_dict['base_pose'][i][0],
                                                  id=0)
                for x in range(10):
                    print ('placement: ', i)
                    visualize_helper.visualize_object(self.samples_dict['object_pose'][i][0],
                                                      filepath="package://config/descriptions/meshes/objects/cylinder_simplify.stl",
                                                      frame_id="proposals_base", name="/object_placement",
                                                      color=[1, 0.5, 0, 1])
                    rospy.sleep(.1)
                if flag=="collisions":
                        for grasp_id in range(len(self.samples_dict['gripper_poses'][i])):
                            gripper_pose_left = self.samples_dict['gripper_poses'][i][grasp_id][0]
                            gripper_pose_right = self.samples_dict['gripper_poses'][i][grasp_id][1]
                            wrist_to_tip = roshelper.list2pose_stamped([0.0, 0.071399, -0.14344421, 0.0, 0.0, 0.0, 1.0],
                                                                       '')
                            wrist_left = roshelper.convert_reference_frame(wrist_to_tip,
                                                                           roshelper.unit_pose(),
                                                                           gripper_pose_left,
                                                                           "proposals_base")
                            wrist_right = roshelper.convert_reference_frame(wrist_to_tip,
                                                                            roshelper.unit_pose(),
                                                                            gripper_pose_right,
                                                                            "proposals_base")
                            collisions = self.samples_dict['collisions'][i][grasp_id]
                            rospy.sleep(.2)

                            if all(collisions[x] == 0 for x in range(len(collisions))):
                                visualize_helper.visualize_object(self.samples_dict['object_pose'][i][0],
                                                                  filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
                                                                  frame_id="proposals_base", name="/object_placement",
                                                                  color=[1, 0.5, 0, 1])
                                visualize_helper.visualize_object(wrist_left,
                                                                  filepath="package://config/descriptions/meshes/mpalm/mpalms_all_coarse.stl",
                                                                  frame_id="proposals_base", name="/gripper_left")
                                visualize_helper.visualize_object(wrist_right,
                                                                  filepath="package://config/descriptions/meshes/mpalm/mpalms_all_coarse.stl",
                                                                  frame_id="proposals_base", name="/gripper_right")
                                rospy.sleep(.2)

        elif flag == "mesh" or flag == "all":
            for id in range(6):
                self.visualize_placement(id, self.samples_dict['points'])
                self.visualize_placement(id, self.collision_free_samples['points'])
                roshelper.handle_block_pose(self.sampler.object_pose_initial,
                                            self.sampler.br,
                                            'proposals_base',
                                            'stable_placement');
                rospy.sleep(.05)

    def remove_edge(self, placement_sequence, placement, grasp_id):
        # if placement[1] == placement_sequence[-1][1]:
        #     print ('edge removed: ', str(placement[1]) + "_" + str(grasp_id[0]), 'end')
        #     self.graph_layer_2.remove_edge(str(placement[1]) + "_" + str(grasp_id[0]), 'end', both_ends=True)
        # else:
        print ('edge removed: ', str(placement[0]) + "_" + str(grasp_id[0]) +  " : " + str(placement[1]) + "_" + str(grasp_id[0]))
        self.graph_layer_2.remove_edge(str(placement[0]) + "_" + str(grasp_id[0]),
                                      str(placement[1]) + "_" + str(grasp_id[0]), both_ends=True)

    def initialize_stable_placements(self):
        """convert samples taken in nominal pose to all stable placements poses"""
        #samples are expressed in proposals_base
        base_pose  = self.sampler.base_initial
        # angle_list = list(np.linspace(np.pi / 4, np.pi / 4 + 2* np.pi, 4))
        angle_list = list(np.linspace(0, 2 * np.pi, 12))
        for placement_id in range(len(self.sampler.object.stable_placement_dict['convex_face_3d'])):
            print ('generating grasps for placement id: ', placement_id)
            grasp_id = 0
            #1. find transformation to stable placement from base (transformation includes rot + trans)
            T = self.sampler.object.stable_placement_dict['T'][placement_id]
            object_pose = roshelper.pose_from_matrix(T,
                                                     frame_id="yumi_body")
            #2. rotate all grasp points, normals, grasp_poses
            grasp_id_list_new = []
            grasp_point_list_new = []
            grasp_normal_list_new = []
            grasp_gripper_pose_list_new = []
            grasp_width_list_new = []
            grasp_T_list_new = []
            grasp_T_pose_list_new = []
            grasp_base_pose_list_new = []
            grasp_object_pose_list_new = []
            grasp_collisions_list = []
            grasp_sample_id_list = []

            #iterate over sample points
            for point_list, normal_list, sample_id in zip(self.dict_grasp['points'], self.dict_grasp['face_normals'], self.dict_grasp['sample_ids']):
                point_list_new = []
                normal_list_new = []
                gripper_pose_list_new = []
                T_list_new = []
                T_pose_list_new = []
                for point, normal in zip(point_list, normal_list):
                    point_new = helper.vector_transform(T, point)
                    T_rot = deepcopy(T)
                    T_rot[:, 3] = [0, 0, 0, 1]
                    normal_new = helper.vector_transform(T_rot, normal)
                    point_list_new.append(point_new)
                    normal_list_new.append(normal_new)
                pose_list, grasp_width, T_gripper_body_list, is_flipped = grasp_from_proposal(point_list, normal_list, T, self.sampler.sampling_pose_initial)
                if is_flipped:
                    tmp = normal_list_new[0]
                    normal_list_new[0] = normal_list_new[1]
                    normal_list_new[1] = tmp
                T_list_new.append([T,T])
                T_pose_list_new.append(T_gripper_body_list)
                for angle in angle_list:
                    #publish gripper reference frame
                    # roshelper.handle_block_pose(pose_list[0], self.sampler.planner.br, 'proposals_base', 'gripper_left_blue')
                    # roshelper.handle_block_pose(pose_list[1], self.sampler.planner.br, 'proposals_base', 'gripper_right_blue')
                    #i. convert to gripper frame
                    gripper_left_gripper_left = roshelper.convert_reference_frame(pose_list[0],
                                                                                  pose_list[0],
                                                                                  roshelper.unit_pose(),
                                                                                  frame_id = "gripper_left")
                    gripper_right_gripper_left = roshelper.convert_reference_frame(pose_list[1],
                                                                                   pose_list[0],
                                                                                   roshelper.unit_pose(),
                                                                                   frame_id="gripper_left")
                    #ii. rotate grippers by angle about y axis of gripper left frame
                    T_pose_rotation_gripper_left = tfm.euler_matrix(0,angle,0, 'rxyz')
                    pose_rotation_gripper_left = roshelper.pose_from_matrix(T_pose_rotation_gripper_left,  'gripper_left')
                    gripper_left_rotated_gripper_left = roshelper.transform_pose(gripper_left_gripper_left, pose_rotation_gripper_left)
                    gripper_right_rotated_gripper_left = roshelper.transform_pose(gripper_right_gripper_left, pose_rotation_gripper_left)
                    #iii. convert back to proposals_base
                    gripper_left_rotated_proposals_base = roshelper.convert_reference_frame(gripper_left_rotated_gripper_left,
                                                                                            roshelper.unit_pose(),
                                                                                            pose_list[0],
                                                                                            frame_id="proposals_base")
                    gripper_right_rotated_proposals_base = roshelper.convert_reference_frame(gripper_right_rotated_gripper_left,
                                                                                            roshelper.unit_pose(),
                                                                                             pose_list[0],
                                                                                   frame_id="proposals_base")
                    #3. check collisions between grippers and table
                    gripper_pose_list_new = [gripper_left_rotated_proposals_base, gripper_right_rotated_proposals_base]
                    collision_list = self.check_collisions(gripper_pose_list_new)
                    grasp_id_list_new.append(placement_id)
                    grasp_point_list_new.append(np.array(point_list_new))
                    grasp_normal_list_new.append(np.array(normal_list_new))
                    grasp_gripper_pose_list_new.append(np.array(gripper_pose_list_new))
                    grasp_width_list_new.append(grasp_width)
                    grasp_T_list_new.append(np.array(T_list_new))
                    grasp_T_pose_list_new.append(np.array(T_pose_list_new))
                    grasp_base_pose_list_new.append(base_pose)
                    grasp_object_pose_list_new.append(object_pose)
                    grasp_collisions_list.append(collision_list)
                    grasp_sample_id_list.append(grasp_id)
                    grasp_id += 1
            self.samples_dict['placement_id'].append(grasp_id_list_new)
            self.samples_dict['points'].append(grasp_point_list_new)
            self.samples_dict['face_normals'].append(grasp_normal_list_new)
            self.samples_dict['gripper_poses'].append(grasp_gripper_pose_list_new)
            self.samples_dict['grasp_width'].append(grasp_width_list_new)
            self.samples_dict['T'].append(grasp_T_list_new)
            self.samples_dict['T_pose'].append(grasp_T_pose_list_new)
            self.samples_dict['base_pose'].append(grasp_base_pose_list_new)
            self.samples_dict['object_pose'].append(grasp_object_pose_list_new)
            self.samples_dict['collisions'].append(grasp_collisions_list)
            self.samples_dict['sample_ids'].append(grasp_sample_id_list)

    def visualize_placement(self, id, proposals=None):
        T_object_world = self.samples_dict['T'][id][0][0][0]
        pose_object_world = roshelper.pose_from_matrix(T_object_world, type="list")
        translated_mesh = self.sampler.object.transform_object(pose_object_world)
        if proposals:
            proposals = proposals[id]
            self.sampler.object.plot_proposals(translated_mesh, proposals)
        else:
            self.sampler.object.plot_meshes([translated_mesh])
        pass

    def check_collisions(self, gripper_pose_list):
        wrist_to_tip = roshelper.list2pose_stamped([0.0, 0.071399, -0.14344421, 0.0, 0.0, 0.0, 1.0], '')
        wrist_left_proposal = roshelper.convert_reference_frame(wrist_to_tip,
                                                       roshelper.unit_pose(),
                                                       gripper_pose_list[0],
                                                       "proposals_base")
        wrist_right_proposal = roshelper.convert_reference_frame(wrist_to_tip,
                                                        roshelper.unit_pose(),
                                                        gripper_pose_list[1],
                                                        "proposals_base")
        wrist_left_world = roshelper.convert_reference_frame(wrist_left_proposal,
                                                       roshelper.unit_pose(),
                                                       self.sampler.base_initial,
                                                       "yumi_body")
        wrist_right_world = roshelper.convert_reference_frame(wrist_right_proposal,
                                                        roshelper.unit_pose(),
                                                        self.sampler.base_initial,
                                                        "yumi_body")
        self.sampler.gripper_left.setCollisionPose(self.sampler.gripper_left.collision_object,
                                           wrist_left_world)

        self.sampler.gripper_right.setCollisionPose(self.sampler.gripper_right.collision_object,
                                            wrist_right_world)
        collision_list = []
        collision_list.append(objects.is_collision(self.sampler.gripper_left.collision_object,
                                                   self.sampler.table.collision_object))
        collision_list.append(objects.is_collision(self.sampler.gripper_right.collision_object,
                                                   self.sampler.table.collision_object))
        # visualize_helper.visualize_object(wrist_left_world,
        #                                   filepath="package://config/descriptions/meshes/mpalm/mpalms_all_coarse.stl",
        #                                   frame_id="yumi_body", name="/gripper_left")
        # visualize_helper.visualize_object(wrist_right_world,
        #                                   filepath="package://config/descriptions/meshes/mpalm/mpalms_all_coarse.stl",
        #                                   frame_id="yumi_body", name="/gripper_right")
        # visualize_helper.visualize_object(roshelper.list2pose_stamped(self.sampler.table_pose, "yumi_body"),
        #                                   filepath="package://config/descriptions/meshes/yumi/table_top_collision.stl",
        #                                   name="/table_pose2", color=(0., 0., 1., 1.),
        #                                   frame_id="/yumi_body")
        # roshelper.handle_block_pose(gripper_pose_list[0], self.sampler.planner.br, "proposals_base", "gripper_left")
        # roshelper.handle_block_pose(gripper_pose_list[1], self.sampler.planner.br, "proposals_base", "gripper_right")
        # roshelper.handle_block_pose(wrist_left, self.sampler.planner.br, "proposals_base", "wrist_left")
        # print ('collision_list: ', collision_list)
        # rospy.sleep(.4)

# wrist_left_world.pose.position.y += 0.05
# wrist_right_world.pose.position.y += 0.05
# self.sampler.planner.gripper_left.setCollisionPose(self.sampler.planner.gripper_left.collision_object,
#                                                    wrist_left_world,
#                                                    self.sampler.planner.listener,
#                                                    type="PoseStamped",
#                                                    frame_id="yumi_body")
# self.sampler.planner.gripper_right.setCollisionPose(self.sampler.planner.gripper_right.collision_object,
#                                                     wrist_right_world,
#                                                     self.sampler.planner.listener,
#                                                     type="PoseStamped",
#                                                     frame_id="yumi_body")
# collision_list = []
# collision_list.append(objects.is_collision(self.sampler.planner.gripper_left.collision_object,
#                                            self.sampler.planner.table.collision_object))
# collision_list.append(objects.is_collision(self.sampler.planner.gripper_right.collision_object,
#                                            self.sampler.planner.table.collision_object))
# visualize_helper.visualize_object(wrist_left_world,
#                                   filepath="package://config/descriptions/meshes/mpalm/mpalms_all_coarse.stl",
#                                   frame_id="yumi_body", name="/gripper_left")
# visualize_helper.visualize_object(wrist_right_world,
#                                   filepath="package://config/descriptions/meshes/mpalm/mpalms_all_coarse.stl",
#                                   frame_id="yumi_body", name="/gripper_right")
# visualize_helper.visualize_object(roshelper.list2pose_stamped(self.sampler.table_pose, "yumi_body"),
#                                   filepath="package://config/descriptions/meshes/yumi/table_top_collision.stl",
#                                   name="/table_pose2", color=(0., 0., 1., 1.),
#                                   frame_id="/yumi_body")
# print (collision_list)
        return collision_list

    def point_far_from_samples(self, point, tol = 0.02):
        if len(self.dict_grasp['points'])==0:
            return True
        for point_list in self.dict_grasp['points']:
            for point_bank in point_list:
                if np.linalg.norm(point - point_bank) < tol:
                    return False
        return True

    def remove_collisions(self):
        filtered_dict = {}
        for key in self.samples_dict.keys():
            filtered_dict[key] = []
            for placement_id in range(len(self.samples_dict['placement_id'])):
                list_tmp = []
                for grasp_id in range(len(self.samples_dict['placement_id'][placement_id])):
                    collisions = self.samples_dict["collisions"][placement_id][grasp_id]
                    if all(collisions[x] == 0 for x in range(len(collisions))):
                        list_tmp.append(self.samples_dict[key][placement_id][grasp_id])
                filtered_dict[key].append(list_tmp)
        self.collision_free_samples = filtered_dict

def gripper_from_proposal( grasp_point_list, grasp_normal_list):
    '''Return gripper pose from sampled grasp points (in sampling_base frame)'''
    dist_grasp = np.linalg.norm(grasp_point_list[0] - grasp_point_list[1])
    if dist_grasp < 0.01:
        print ('[Grasping] Distance between grasp is small (<0.01). Something is wrong.')
    pose_list = []
    for point, normal in zip(grasp_point_list, grasp_normal_list):
        y_vec = normal
        if normal[0]==0 and normal[1]==0:
            x_vec = np.array([1, 0, 0])
        else:
            x_vec = np.array([0,0,-1])
        z_vec = np.cross(x_vec, y_vec)
        x_vec = x_vec / np.linalg.norm(x_vec)
        y_vec = y_vec / np.linalg.norm(y_vec)
        z_vec = z_vec / np.linalg.norm(z_vec)
        # Normalized object frame
        hand_orient_norm = np.vstack((x_vec, y_vec, z_vec))
        hand_orient_norm = hand_orient_norm.transpose()
        quat = roshelper.mat2quat(hand_orient_norm)
        #define hand pose
        pose_list.append(roshelper.list2pose_stamped(list(point) + list(quat)))

    return pose_list, dist_grasp  #[left, right]

def grasp_from_proposal(point_list_new, normal_list_new, T, sampling_base_pose):
    '''Returns gripper pose in proposals base frame from sampled grasp points and normals'''
    #point_list: in free object frame (sampling base)
    #point_normal: in free object frame (sampling base)
    #T: transformation matrix from sampling base to object stable placement

    pose_list, grasp_width = gripper_from_proposal(point_list_new, normal_list_new)
    pose_stamped_list = []
    for pose in pose_list:
        # pose_stamped_world = self.manipulation_3d.planner.listener.transformPose("yumi_body",
        #                                                  pose)
        pose_stamped_world = roshelper.convert_reference_frame(pose,
                                                               roshelper.unit_pose(),
                                                               sampling_base_pose,
                                                                "yumi_body")
        pose_stamped_list.append(pose_stamped_world)
    pose_stamped_list_sorted, is_flipped = sort_pose_list(pose_stamped_list)
    pose_stamped_list_aligned = align_arm_poses(pose_stamped_list_sorted[0], grasp_width)
    pose_new_list = []
    T_pose_list = []
    for gripper_pose_world in pose_stamped_list_aligned:
        #convert to local proposals frame
        # gripper_pose_body = self.manipulation_3d.planner.listener.transformPose("sampling_base",
        #                                                                         gripper_pose_world)
        gripper_pose_body = roshelper.convert_reference_frame(gripper_pose_world,
                                                               sampling_base_pose,
                                                              roshelper.unit_pose(),
                                                              "sampling_base")
        T_gripper_body = roshelper.matrix_from_pose(gripper_pose_body)
        #rotate pose by T (in sampling_base frame)
        #there is a sublelty here:
        #T convert from sampling base to proposals frame
        T_gripper_body_new = np.matmul(T, T_gripper_body)
        pose_new_list.append(roshelper.pose_from_matrix(T_gripper_body_new,
                                                        frame_id="proposals_base"))
        T_pose_list.append(T_gripper_body)

    return pose_new_list, grasp_width, T_pose_list, is_flipped

def sort_pose_list(pose_list):
    '''Identify point to be grasped by left arm'''
    is_flipped = False
    if pose_list[0].pose.position.y < pose_list[1].pose.position.y:
        pose_list_new = [pose_list[1], pose_list[0]]
        is_flipped = True
    else:
        pose_list_new = pose_list
    return pose_list_new, is_flipped

def align_arm_poses(pose, grasp_dist):
    '''Set position of right arm with respect to left arm'''
    T_left_world = roshelper.matrix_from_pose(pose)
    pose_right_left = PoseStamped()
    pose_right_left.header.frame_id = pose.header.frame_id
    pose_right_left.pose.position.y = - grasp_dist
    quat = roshelper.rotate_quat_y(pose_right_left)
    pose_right_left.pose.orientation.x = quat[0]
    pose_right_left.pose.orientation.y = quat[1]
    pose_right_left.pose.orientation.z = quat[2]
    pose_right_left.pose.orientation.w = quat[3]
    T_right_left = roshelper.matrix_from_pose(pose_right_left)
    T_right_world = np.matmul(T_left_world, T_right_left)
    pose_right = roshelper.pose_from_matrix(T_right_world)
    return [pose, pose_right]
