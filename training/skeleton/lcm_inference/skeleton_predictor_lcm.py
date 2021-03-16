import os, os.path as osp
import copy
import time
import numpy as np
import sys
import lcm
from scipy.spatial.transform import Rotation as R

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
import torch.nn as nn

from airobot import set_log_level, log_debug, log_info, log_warn, log_critical

sys.path.append('..')
from skeleton_utils.skeleton_globals import SOS_token, EOS_token, PAD_token
from skeleton_utils.utils import process_pointcloud_batch
from skeleton_utils.glamor_utils import get_uniform_sample_inds, get_boltzmann_sample_inds, filter_skills_probs


import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
from rpo_lcm import point_cloud_t, pose_stamped_t, string_array_t, rpo_plan_skeleton_t
from rpo_planning.utils import common as util
from rpo_planning.utils import lcm_utils


class GlamorSkeletonPredictorLCM():
    def __init__(self, lc, model, prior_model, inverse_model, args, language, model_path=None,
                 pcd_sub_name='explore_pcd_obs', task_sub_name='explore_task_obs', 
                 skeleton_pub_name='explore_skill_skeleton', verbose=False):
        """
        Constructor for GlamorSkeletonPredictorLCM, which is used to predict plan skeletons
        using a trained neural network and communicate with the robot environment process
        using LCM.

        Args:
            lc (lc.LCM): Main LCM handle
            model (torch.nn.Module): Trained neural network that will make the skeleton predictions
            inverse_model (torch.nn.Module): Trained task encoder neural network which produces embeddings
                for the start pcd, goal pcd, and desired transformation. # TODO: combine model and inverse model
            args (argparse.Namespace): Arguments that were used when training the NN, which may
                contain useful parameters to use during prediction.
            model_path (str): Path to model weights which were loaded (UNUSED AT THE MOMENT) 
            language (skeleton_utils.language.SkillLanguage): Contains mapping from categorical indices to 
                strings contained in the language
            pcd_sub_name (str, optional): Name of incoming LCM message containing the point cloud observations
                from the environment. Defaults to 'explore_pcd_obs'.
            task_sub_name (str, optional): Name of incoming LCM message containing the desired transformation
                specification. Defaults to 'explore_task_obs'.
            skeleton_pub_name (str, optional): Name of outgoing LCM message containing the predicted skeleton
               . Defaults to 'explore_skill_skeleton'.
        """
        self.model = model
        self.inverse_model = inverse_model
        self.prior_model = prior_model
        self.args = args
        self.model_path = model_path
        self.language = language

        self.pcd_sub_name = pcd_sub_name
        self.task_sub_name = task_sub_name
        self.skeleton_pub_name = skeleton_pub_name

        # self.lc = lcm.LCM()
        self.lc = lc
        self.pcd_sub = self.lc.subscribe(self.pcd_sub_name, self.pcd_sub_handler)
        self.task_sub = self.lc.subscribe(self.task_sub_name, self.task_sub_handler)

        self.verbose = verbose

    def pcd_sub_handler(self, channel, data):
        msg = point_cloud_t.decode(data)
        points = msg.points
        num_pts = msg.num_points

        self.points = lcm_utils.unpack_pointcloud_lcm(points, num_pts)
        self.received_pcd_data = True

    def task_sub_handler(self, channel, data):
        msg = pose_stamped_t.decode(data)
        self.task_pose_list = lcm_utils.pose_stamped2list(msg)
        self.received_task_data = True

    def prepare_model_inputs(self, points, transformation_des):
        """
        Function to prepare the point cloud observation and task encoding we got
        from LCM to pass into the NN. This function converts the data types, 
        obtains a final goal point cloud, and returns the variables that can 
        be directly inputted to the NN.

        Args:
            points (list): List of lists, each list is [x, y, z] point coordinate. `
            transformation_des (list): 6D pose indicating desired relative transformation
                of the point cloud. 
        """
        # convert to numpy and get goal point cloud
        start_pcd_np = np.asarray(points)
        transformation_des_np = np.asarray(transformation_des)
        
        transformation_des_mat = np.eye(4)
        transformation_des_mat[:-1, -1] = transformation_des[:3]
        transformation_des_mat[:-1, :-1] = R.from_quat(transformation_des[3:]).as_matrix()

        goal_pcd_np = np.matmul(
            transformation_des_mat,
            np.concatenate((start_pcd_np, np.ones((start_pcd_np.shape[0], 1))), axis=1).T
        )[:-1, :].T
        
        # convert to torch tensor
        observation = torch.from_numpy(start_pcd_np).float()
        next_observation = torch.from_numpy(goal_pcd_np).float()
        subgoal = torch.from_numpy(transformation_des_np).float()
        return observation, next_observation, subgoal

    def predict_skeleton(self, N=50, max_steps=5, sampling='boltzmann'):
        model = self.model
        inverse_model = self.inverse_model
        prior_model = self.prior_model
        args = self.args
        language = self.language

        if args.cuda:
            dev = torch.device('cuda:0')
        else:
            dev = torch.device('cpu')

        model.eval().to(dev)
        inverse_model.eval().to(dev)
        prior_model.eval().to(dev)

        with torch.no_grad():
            # process incoming data from LCM
            self.received_pcd_data = False
            self.received_task_data = False
            while True: 
                self.lc.handle_timeout(10*1e3)
                if self.received_task_data and self.received_pcd_data:
                    break
                if self.verbose:
                    log_debug('Predict skeleton: handler time out after 10s, looping once more')
            
            points = self.points
            transformation_des = self.task_pose_list

            # process this for neural network input
            observation, next_observation, subgoal = self.prepare_model_inputs(points, transformation_des)

            observation = observation.float().to(dev)[None, :, :]
            next_observation = next_observation.float().to(dev)[None, :, :]
            subgoal = subgoal.float().to(dev)[None, :]

            observation = process_pointcloud_batch(observation)
            next_observation = process_pointcloud_batch(next_observation)

            # use inverse model to get overall task encoding
            task_emb = inverse_model(observation, next_observation, subgoal)
            prior_emb = inverse_model.prior_forward(observation)

            # predict skeleton, up to max length
            decoder_input = torch.Tensor([[SOS_token]]).long().to(dev)
            decoder_hidden = task_emb[None, :]
            p_decoder_input = torch.Tensor([[SOS_token]]).long().to(dev)
            p_decoder_hidden = prior_emb[None, :]

            # sample uniformly for skill sequences
            if sampling not in ['boltzmann', 'uniform']:
                sampling = 'boltzmann'
            if sampling == 'uniform':
                skill_inds = get_uniform_sample_inds(len(language.index2skill.keys()), N=N, max_steps=max_steps)
                skill_inds = torch.from_numpy(skill_inds).long().to(dev)
            else:
                skill_inds = get_boltzmann_sample_inds(model, prior_model, task_emb, prior_emb, N=N, max_steps=max_steps)
            seq_to_score = torch.cat((decoder_input.repeat(skill_inds.size(0), 1), skill_inds), axis=1)
            p_seq_to_score = torch.cat((p_decoder_input.repeat(skill_inds.size(0), 1), skill_inds), axis=1)

            # forward pass to get embeddings for each sequence
            seq_embed = model.embed(seq_to_score)
            p_seq_embed = prior_model.embed(p_seq_to_score)

            decoder_output, _ = model.gru(seq_embed, decoder_hidden.repeat(1, skill_inds.size(0), 1))
            p_decoder_output, _ = prior_model.gru(p_seq_embed, p_decoder_hidden.repeat(1, skill_inds.size(0), 1))

            # get logits from output embeddings
            output_logprobs = model.log_softmax(model.out(decoder_output).permute(0, 2, 1)).permute(0, 2, 1)
            p_output_logprobs = prior_model.log_softmax(prior_model.out(p_decoder_output).permute(0, 2, 1)).permute(0, 2, 1)
            output_logprobs_g = torch.gather(output_logprobs[:, 1:, :], -1, skill_inds[:, :, None]).squeeze()
            p_output_logprobs_g = torch.gather(p_output_logprobs[:, 1:, :], -1, skill_inds[:, :, None]).squeeze()
            
            # filter any that we know are bad
            filtered_skills, output_logprobs_g, p_output_logprobs_g = filter_skills_probs(
                skill_inds,
                output_logprobs_g,
                p_output_logprobs_g,
                logprob_thresh=args.logprob_thresh
            )

            output_probs, p_output_probs = torch.exp(output_logprobs_g), torch.exp(p_output_logprobs_g)

            # all_valid_skills = []
            # for i in range(filtered_skills.size(0)):
            #     ind_seq = filtered_skills[i]
            #     seq = []
            #     for j in ind_seq:
            #         seq.append(language.index2skill[j.item()])
            #     all_valid_skills.append(seq)

            if output_probs.size(0) > 0:
                # get likelihood ratios and score
                # compute product of the likelihoods
                inverse_prod, prior_prod = output_probs.prod(-1), p_output_probs.prod(-1)

                # ratio_objective = decoder_output[:, -1] / p_decoder_output[:, -1]
                ratio_objective = inverse_prod / prior_prod
                ratio_argmax = ratio_objective.topk(1, dim=0)[-1].item()
                inverse_argmax = inverse_prod.topk(1, dim=0)[-1].item()
                
                best_seq_ratio = filtered_skills[ratio_argmax, :].squeeze()
                best_seq_inverse = filtered_skills[inverse_argmax, :].squeeze()
                decoded_skills_ratio = []
                decoded_skills_inverse = []
                for t in range(best_seq_ratio.size(0)):
                    decoded_skills_ratio.append(language.index2skill[best_seq_ratio[t].item()])
                for t in range(best_seq_inverse.size(0)):
                    decoded_skills_inverse.append(language.index2skill[best_seq_inverse[t].item()])
                ratio_msg = 'Glamor skeleton predictor: decoded skills (ratio): %s' % ', '.join(decoded_skills_ratio)
                inverse_msg = 'Glamor skeleton predictor: decoded skills (inverse): %s' % ', '.join(decoded_skills_inverse)
                if self.verbose:
                    log_debug(ratio_msg)
                    log_debug(inverse_msg)
                decoded_skills = decoded_skills_ratio
                decoded_skill_labels = best_seq_ratio.cpu().numpy().tolist()
            else:
                log_warn('Glamor skeleton predictor: could not sample skill sequence from candidates, defaulting to unrolling ML sequence')
                decoded_skills = []
                decoded_skill_labels = []
                for t in range(args.max_seq_length):
                    # get predictions from model that takes both start and goal
                    decoder_input = model.embed(decoder_input)
                    decoder_output, decoder_hidden = model.gru(decoder_input, decoder_hidden)
                    output = model.log_softmax(model.out(decoder_output[:, 0]))
                    topv, topi = output.topk(1, dim=1)
                    decoder_input = topi            

                    # rank outputs
                    if topi.item() == language.skill2index['EOS']:
                        decoded_skills.append('EOS')
                        decoded_skill_labels.append(topi.item())
                        break
                    else:
                        decoded_skills.append(language.index2skill[topi.item()])
                        decoded_skill_labels.append(topi.item())

            predicted_skeleton = decoded_skills
            if predicted_skeleton[-1] != 'EOS':
                predicted_skeleton.append('EOS')

            skeleton_msg = rpo_plan_skeleton_t()
            skeleton_msg.skill_names.num_strings = len(predicted_skeleton)
            skeleton_msg.skill_names.string_array = predicted_skeleton
            skeleton_msg.num_skills = len(decoded_skill_labels)
            skeleton_msg.skill_indices = decoded_skill_labels

            self.lc.publish(self.skeleton_pub_name, skeleton_msg.encode())
            return True  # TODO: include return flag or other error that can be raised if something goes wrong 
