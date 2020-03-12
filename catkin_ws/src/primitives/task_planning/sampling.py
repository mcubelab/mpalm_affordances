import sys, os
sys.path.append(os.environ['CODE_BASE'] + '/catkin_ws/src/config/src')
from helper import roshelper, helper
import itertools
import numpy as np
import rospy
import util
from copy import deepcopy
from graph import Graph

class Sampling(object):
    def __init__(self, q0, _object, table, gripper_left, gripper_right, listener=None, br=None):
        self.q0 = q0
        self.object = _object
        self.table = table
        self.gripper_left = gripper_left
        self.gripper_right = gripper_right
        self.listener = listener
        self.br = br
        self.initialize_reference_frames(q0)

    def initialize_reference_frames(self, q0):
        q0_planar,stable_placement_object = util.get2dpose_object(q0,self.object)

        sampling_base_pose = util.get3dpose_object(q0_planar,
                                                        self.object,
                                                        stable_placement=0)
        self.object_pose_initial = q0
        self.sampling_pose_initial = sampling_base_pose
        self.base_initial = deepcopy(self.sampling_pose_initial)
        self.base_initial.pose.position.z = 0
        for i in range(10):
            roshelper.handle_block_pose(self.sampling_pose_initial, self.br, 'yumi_body', 'sampling_base')
            roshelper.handle_block_pose(self.base_initial, self.br, 'yumi_body', 'proposals_base')

def build_graph_grasp(sampling, placement_list):
    intersection_dict = connect_grasps(sampling.collision_free_samples)
    graph_layer_1 = build_graph_placements(intersection_dict)
    placement_sequence = graph_layer_1.dijkstra(str(placement_list[0]), str(placement_list[1]))
    graph_layer_2 = build_graph_regrasps(intersection_dict, placement_sequence)
    graph_layer_2 = add_boundary_edges(placement_list, graph_layer_2)
    return graph_layer_2

def build_graph_lever(sampling, placement_list):
    intersection_dict = connect_levers(sampling.samples_dict)
    graph_layer_1 = build_graph_placements(intersection_dict)
    placement_sequence = graph_layer_1.dijkstra(str(placement_list[0]), str(placement_list[1]))
    return placements_from_nodes_lever(placement_sequence)

def build_graph(lever_samples, grasp_samples, placement_list):
    intersection_dict_lever = connect_levers(lever_samples.samples_dict)
    intersection_dict_grasp = connect_grasps(grasp_samples.collision_free_samples)
    intersection_dict_total = helper.merge_two_dicts(intersection_dict_lever, intersection_dict_grasp)
    graph_layer_1 = build_graph_placements(intersection_dict_total)
    graph_layer_1 = add_boundary_edges(placement_list, graph_layer_1)
    placement_sequence = graph_layer_1.dijkstra('start', 'end')
    return placements_from_nodes_lever(placement_sequence)

def search_placement_graph(lever_samples=None, grasp_samples=None, placement_list=None):
    if lever_samples==None:
        intersection_dict_total = connect_grasps(grasp_samples.collision_free_samples)
    elif grasp_samples==None:
        intersection_dict_total = connect_levers(lever_samples.samples_dict)
    else:
        intersection_dict_lever = connect_levers(lever_samples.samples_dict)
        intersection_dict_grasp = connect_grasps(grasp_samples.collision_free_samples)
        intersection_dict_total = helper.merge_two_dicts(intersection_dict_lever,
                                                     intersection_dict_grasp)

    graph_layer_1 = build_graph_placements(intersection_dict_total)
    graph_layer_1 = add_boundary_edges(placement_list, graph_layer_1)
    # from IPython import embed
    # embed()
    placement_sequence = graph_layer_1.dijkstra('start', 'end')
    if lever_samples==None:
        # intersection_dict_total = connect_grasps(grasp_samples.collision_free_samples)
        return placement_sequence, intersection_dict_total
    elif grasp_samples==None:
        # intersection_dict_total = connect_levers(lever_samples.samples_dict)
        return placement_sequence, intersection_dict_total
    else:
        return placement_sequence, intersection_dict_grasp, intersection_dict_lever

def search_primitive_graph(_node_sequence, intersection_dict_grasp=None, intersection_dict_lever=None):
    placement_sequence, primitive_sequence = nodes_to_placement_sequence(_node_sequence)
    sample_sequence = []
    for counter, primitive in enumerate(primitive_sequence):
        if primitive=="levering":
            sample_id_list = [find_common_sample(intersection_dict_lever, placement_sequence[counter], is_flip=False)]
        elif primitive=="grasping":
            sample_id_list = find_common_sample(intersection_dict_grasp, placement_sequence[counter], is_flip=True)
        sample_sequence.append(sample_id_list)
    return placement_sequence, sample_sequence, primitive_sequence

def connect_grasps(grasp_dict):
    intersection_dict = {}
    intersection_dict['placements'] = []
    intersection_dict['sample_ids'] = []
    intersection_dict['is_connected'] = []
    intersection_dict['primitive'] = []
    intersection_dict['cost'] = []
    intersection_dict['num_placements'] = len(grasp_dict['placement_id'])
    intersection_dict['sample_ids_individual'] = grasp_dict['sample_ids']
    for placement_id_1 in range(len(grasp_dict['placement_id'])): #loop through all stable placements
        for placement_id_2 in range(placement_id_1, len(grasp_dict['placement_id'])):
            intersection_list = helper.intersection(grasp_dict["sample_ids"][placement_id_1], grasp_dict["sample_ids"][placement_id_2])
            if len(intersection_list)==0:
                is_connected = False
            else:
                is_connected = True
            intersection_dict['placements'].append([placement_id_1, placement_id_2])
            intersection_dict['sample_ids'].append(intersection_list)
            intersection_dict['primitive'].append('grasping')
            intersection_dict['cost'].append(rospy.get_param('grasping/cost'))
            intersection_dict['is_connected'].append(is_connected)
    return intersection_dict

def connect_levers(lever_dict):
    import lever_sampling
    intersection_dict = {}
    intersection_dict['placements'] = []
    intersection_dict['sample_ids'] = []
    intersection_dict['primitive'] = []
    intersection_dict['cost'] = []
    intersection_dict['is_connected'] = []
    intersection_dict['num_placements'] = len(lever_dict['placement_id'])
    intersection_dict['sample_ids_individual'] = lever_dict['sample_ids']
    for placement_id_1 in range(len(lever_dict['placement_id'])): #loop through all stable placements
        for placement_id_2 in range(0, len(lever_dict['placement_id'])):
            if placement_id_2 in lever_dict['placement_end_id'][placement_id_1]:
                is_connected = True
                index_list, sample_id_list = lever_sampling.identify_placement_ids(lever_dict, [placement_id_1, placement_id_2])
                neighboor_id = index_list[0]
                intersection_dict['sample_ids'].append(lever_dict['sample_ids'][placement_id_1][neighboor_id])
            else:
                is_connected = False
                intersection_dict['sample_ids'].append([])
            intersection_dict['placements'].append([placement_id_1, placement_id_2])
            intersection_dict['is_connected'].append(is_connected)
            intersection_dict['primitive'].append('levering')
            intersection_dict['cost'].append(rospy.get_param('levering/cost'))
    return intersection_dict

def build_graph_placements(intersection_dict):
    edge_list = []
    for counter, connections in enumerate(intersection_dict['placements']):
        if intersection_dict['is_connected'][counter]:
            if intersection_dict['primitive'][counter]=='grasping':
                is_both_directions = True
            else:
                is_both_directions = False
            edge_list.append((str(connections[0]) + "_" + intersection_dict['primitive'][counter], str(connections[1]) + "_" + intersection_dict['primitive'][counter],intersection_dict['cost'][counter], is_both_directions))
    return Graph(edge_list)

def search_graph(sampling, placement_list):
    graph_layer_2 = build_graph(sampling, placement_list)
    node_sequence = graph_layer_2.dijkstra("start", "end"); print ('node_sequence: ', node_sequence)
    placement_sequence, grasp_id_sequence = placements_from_nodes(node_sequence, placement_list)
    return placement_sequence, grasp_id_sequence

def find_common_sample(intersection_dict, placement_list, is_flip=True):
    for index, term in enumerate(intersection_dict['placements']):
        if is_flip:
            if term == placement_list or term == [placement_list[1], placement_list[0]]:
                break
        else:
            if term == placement_list:
                break
    return intersection_dict['sample_ids'][index]


def build_graph_regrasps(intersection_dict, placement_sequence):
    placement_sequence = np.array(placement_sequence).astype(int)
    placement_index_sequence = []
    placement_link_list = []
    grasp_list = []
    node_list = []
    edge_list = []
    for counter in range(len(placement_sequence) - 1):
        placement_link = [placement_sequence[counter], placement_sequence[counter + 1]]
        for index, term in enumerate(intersection_dict['placements']):
            if term == placement_link or term == [placement_link[1], placement_link[0]]:
                placement_index_sequence.append(index)
                break
        grasp_list.append(intersection_dict['sample_ids'][index])
        placement_link_list.append(placement_link)
        # add nodes for regrasps within placement id
        for placement in placement_link:
            for sample_id in intersection_dict['sample_ids'][index]:
                node_list.append([placement, sample_id])
    # build edges for regrasps within placement id
    placement_node_list = []
    for placement in placement_sequence:
        placement_node_list_tmp = []
        for node in node_list:
            if node[0] == placement:
                placement_node_list_tmp.append(node[1])
        placement_node_list.append(placement_node_list_tmp)
        within_placement_connections = list(itertools.combinations(intersection_dict['sample_ids_individual'][placement], 2))
        for connection in within_placement_connections:
            edge_list.append(
                [str(placement) + "_" + str(connection[0]), str(placement) + "_" + str(connection[1]), 1, True])
    for index, placements in zip(placement_index_sequence, placement_link_list):
        grasp_ids = intersection_dict['sample_ids'][index]
        for connection in grasp_ids:
            edge_list.append(
                [str(placements[0]) + "_" + str(connection), str(placements[1]) + "_" + str(connection), 1,
                 True])
    return Graph(edge_list)

def add_boundary_edges(placement_list, graph):
    edges = graph.get_edges()
    for edge in edges:
        #1. extract
        start_placement_node = edge[0].split('_')[0]
        end_placement_node = edge[1].split('_')[0]
        #1. check if stable placement is associated with
        if 'grasping' in edge[0]:
            cost = rospy.get_param('/grasping/cost')
        elif 'levering' in edge[0]:
            cost = rospy.get_param('/levering/cost')
        if start_placement_node == str(placement_list[0]):
            graph.add_edge("start", edge[0], cost, False)
        if end_placement_node == str(placement_list[1]):
            graph.add_edge(edge[1], "end", cost, False)
    return graph

def placements_from_graph_nodes(_node_sequence, placement_list):
    node_sequence = list(_node_sequence)[1:-1]
    placement_sequence = []
    grasp_id_sequence = []
    for counter, node in enumerate(node_sequence):
        node_previous = node_sequence[counter - 1]
        node_name_previous = node_previous.split('_')
        node_name = node.split('_')
        if node=='start':
            pass
        elif node=='end':
            placement_sequence.append(int(placement_list[1]))
            grasp_id_sequence.append(int(node_name_previous[1]))
        else:
            placement_sequence.append(int(node_name[0]))
            grasp_id_sequence.append(int(node_name[1]))
    placement_sequence = zip(*[placement_sequence[i::2] for i in range(2)])
    grasp_id_sequence = zip(*[grasp_id_sequence[i::2] for i in range(2)])

    return placement_sequence, grasp_id_sequence

def placements_from_nodes_lever(placement_sequence_tuple):
    placement_sequence = [int(list(placement_sequence_tuple)[x]) for x in range(len(list(placement_sequence_tuple)))]
    placement_sequence_new = []
    for i in range(len(placement_sequence)-1):
        counter = i
        placement_sequence_new.append([placement_sequence[counter], placement_sequence[counter+1]])
    return placement_sequence_new

def nodes_to_placement_sequence(_node_sequence_list):
    node_sequence = list(_node_sequence_list)[1:-1]
    placement_sequence_new = []
    primitive_sequence_new = []
    for i in range(len(node_sequence)-1):
        counter = i
        node_initial = node_sequence[i]
        node_final = node_sequence[i+1]
        node_name_start = node_initial.split('_')
        node_name_end = node_final.split('_')
        placement_start = node_name_start[0]
        placement_end = node_name_end[0]
        primitive = node_name_start[1]
        placement_sequence_new.append([int(placement_start), int(placement_end)])
        primitive_sequence_new.append(primitive)
    return placement_sequence_new, primitive_sequence_new
