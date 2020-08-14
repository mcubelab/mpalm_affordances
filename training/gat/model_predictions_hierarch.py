import torch
import torch.nn.functional as F
import os
import torch.multiprocessing as mp
import numpy as np
import os.path as osp
from torch.utils.data import DataLoader
# from easydict import EasyDict
from torch.nn.utils import clip_grad_norm
import pdb
import torch.nn as nn

# from data import RobotKeypointsDatasetGrasp
from random import choices
# from apex.optimizers import FusedAdam
from torch.optim import Adam
import argparse
from itertools import permutations
import itertools
import matplotlib.pyplot as plt
import sys, signal

sys.path.append('/root/training/gat_vq/')
from models_vae_h import GoalVAE, PalmVAE, GeomPalmVAE, GeomGoalVAE, GoalPointVAE, PalmPointVAE
import time
import copy

sys.path.append('/root/catkin_ws/src/primitives/')
from eval_utils.model_test_tools import ModelEvaluator


"""Parse input arguments"""
parser = argparse.ArgumentParser(description='Train reasoning model')
parser.add_argument('--train', action='store_true', help='whether or not to train')
parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not')

# Generic Parameters for Experiments
parser.add_argument('--dataset', default='intphys', type=str, help='intphys or others')
parser.add_argument('--logdir', default='vae_cachedir', type=str, help='location where log of experiments will be stored')
parser.add_argument('--exp', default='debug', type=str, help='name of experiments')
parser.add_argument('--data_workers', default=16, type=int, help='Number of different data workers to load data in parallel')

# train
parser.add_argument('--batch_size', default=64, type=int, help='size of batch of input to use')
parser.add_argument('--num_epoch', default=10000, type=int, help='number of epochs of training to run')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for training')
parser.add_argument('--log_interval', default=10, type=int, help='log outputs every so many batches')
parser.add_argument('--save_interval', default=1000, type=int, help='save outputs every so many batches')
parser.add_argument('--resume_iter', default=0, type=str, help='iteration to resume training')

parser.add_argument('--vis', action='store_true', help='vis')

# Distributed training hyperparameters
parser.add_argument('--nodes', default=1, type=int, help='number of nodes for training')
parser.add_argument('--gpus', default=1, type=int, help='number of gpus per nodes')
parser.add_argument('--node_rank', default=0, type=int, help='rank of node')
parser.add_argument('--latent_dimension', type=int,
                    default=256)
parser.add_argument('--kl_anneal_rate', type=float, default=0.9999)
parser.add_argument('--prediction_dir', type=str, default='/tmp/predictions')
parser.add_argument('--observation_dir', type=str, default='/tmp/observations')
parser.add_argument('--pointnet', action='store_true')

parser.add_argument('--primitive_name', type=str, default='grasp')
parser.add_argument('--subgoal', type=str, default='mask')
parser.add_argument('--H2', action='store_true')
parser.add_argument('--indep', action='store_true')

def test_joint(goal_eval, contact_eval, model, goal_model, obs_file, FLAGS, model_path):
    print('Making prediction, loading observation from file: ' + obs_file)
    model = model.eval()
    vae = model
    counter = 0

    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    vae = vae.to(dev)

    with torch.no_grad():
        # for start, transformation, obj_frame, kd_idx, vis_info, decoder_x, obj_id in test_dataloader:
        while True:
            try:
                observation = np.load(osp.join(FLAGS.observation_dir, obs_file), allow_pickle=True)
                break
            except:
                pass
            time.sleep(0.01)

        data = {}
        data['start'] = observation['pointcloud_pts']
        subgoal_out = goal_eval.vae_eval_goal_both(data)
        masks, trans = subgoal_out[0], subgoal_out[1]
        # masks = goal_eval.vqvae_eval_goal(data)

        preds_all = []
        for ind in range(masks.shape[0]):
            new_data = copy.deepcopy(data)

            top_inds = np.argsort(masks[ind, :])[::-1]
            pred_mask = np.zeros((masks.shape[1]), dtype=bool)
            pred_mask[top_inds[:15]] = True
            new_data['object_mask_down'] = pred_mask
            new_data['transformation'] = trans[0, :]

            output = contact_eval.vae_eval(new_data)
            # preds = output[0]
            preds = output[0]
            preds_all.append(preds.tolist())
        preds_all = np.asarray(preds_all)
        palm_predictions = preds_all.squeeze()
        mask_predictions = masks.squeeze()
        trans_predictions = trans.squeeze()
        # print(palm_predictions.shape, mask_predictions.shape, trans_predictions.shape)        

        # predictions = np.asarray(predictions).squeeze()
        output_file = osp.join(FLAGS.prediction_dir, obs_file)
        # print('Saving to: ' + str(output_file))
        np.savez(
            output_file,
            palm_predictions=palm_predictions,
            mask_predictions=mask_predictions,
            trans_predictions=trans_predictions,
            model_path=model_path)
        os.remove(osp.join(FLAGS.observation_dir, obs_file))
        print('Saved prediction to: ' + str(output_file))


def test_joint2(goal_eval, contact_eval, model, goal_model, obs_file, FLAGS, model_path):
    print('Making prediction, loading observation from file: ' + obs_file)
    model = model.eval()
    vae = model
    counter = 0

    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    vae = vae.to(dev)

    with torch.no_grad():
        # for start, transformation, obj_frame, kd_idx, vis_info, decoder_x, obj_id in test_dataloader:
        while True:
            try:
                observation = np.load(osp.join(FLAGS.observation_dir, obs_file), allow_pickle=True)
                break
            except:
                pass
            time.sleep(0.01)

        data = {}
        data['start'] = observation['pointcloud_pts']
        output = contact_eval.vae_eval(data, first=True)
        preds = output[0]
        palm_ind = np.random.randint(99)
        palm_pred = preds[palm_ind, :]

        new_data = copy.deepcopy(data)
        new_data['contact_world_frame_right'] = palm_pred[:7]
        new_data['contact_world_frame_left'] = palm_pred[7:]
        subgoal_out = goal_eval.vae_eval_goal_both(new_data, first=False)
        masks, trans = subgoal_out[0], subgoal_out[1]

        preds = np.asarray(preds)
        palm_predictions = preds.squeeze()[None, :]
        mask_predictions = masks.squeeze()[0, :][None, :]
        trans_predictions = trans.squeeze()[0, :][None, :]
        print(palm_predictions.shape, mask_predictions.shape, trans_predictions.shape)        

        # predictions = np.asarray(predictions).squeeze()
        output_file = osp.join(FLAGS.prediction_dir, obs_file)
        # print('Saving to: ' + str(output_file))
        np.savez(
            output_file,
            palm_predictions=palm_predictions,
            mask_predictions=mask_predictions,
            trans_predictions=trans_predictions,
            model_path=model_path)
        os.remove(osp.join(FLAGS.observation_dir, obs_file))
        print('Saved prediction to: ' + str(output_file))


def test_indep(goal_eval, contact_eval, model, goal_model, obs_file, FLAGS, model_path):
    print('Making prediction, loading observation from file: ' + obs_file)
    model = model.eval()
    vae = model
    counter = 0

    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    vae = vae.to(dev)

    with torch.no_grad():
        # for start, transformation, obj_frame, kd_idx, vis_info, decoder_x, obj_id in test_dataloader:
        while True:
            try:
                observation = np.load(osp.join(FLAGS.observation_dir, obs_file), allow_pickle=True)
                break
            except:
                pass
            time.sleep(0.01)

        data = {}
        data['start'] = observation['pointcloud_pts']
        output = contact_eval.vae_eval(data, first=True, pointnet=FLAGS.pointnet)
        preds = output[0]
        if not FLAGS.pointnet:
            palm_ind = np.random.randint(99)
            palm_pred = preds[palm_ind, :]
        else:
            palm_pred = preds

        new_data = copy.deepcopy(data)
        new_data['contact_world_frame_right'] = palm_pred[:7]
        new_data['contact_world_frame_left'] = palm_pred[7:]
        subgoal_out = goal_eval.vae_eval_goal_both(new_data, first=True, pointnet=FLAGS.pointnet)
        masks, trans = subgoal_out[0], subgoal_out[1]

        preds = np.asarray(preds)
        palm_predictions = preds.squeeze()[None, :]
        mask_predictions = masks.squeeze()[0, :][None, :]
        trans_predictions = trans.squeeze()[0, :][None, :]
        print(palm_predictions.shape, mask_predictions.shape, trans_predictions.shape)        

        # predictions = np.asarray(predictions).squeeze()
        output_file = osp.join(FLAGS.prediction_dir, obs_file)
        # print('Saving to: ' + str(output_file))
        np.savez(
            output_file,
            palm_predictions=palm_predictions,
            mask_predictions=mask_predictions,
            trans_predictions=trans_predictions,
            model_path=model_path)
        os.remove(osp.join(FLAGS.observation_dir, obs_file))
        print('Saved prediction to: ' + str(output_file))


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


def main_single(rank, FLAGS):
    rank_idx = FLAGS.node_rank * FLAGS.gpus + rank
    world_size = FLAGS.nodes * FLAGS.gpus

    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1492', world_size=world_size, rank=rank_idx)

    if FLAGS.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")



    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    if not osp.exists(logdir):
        os.makedirs(logdir)
    # logger = TensorBoardOutputFormat(logdir)

    ## dataset
    # dataset_train = RobotKeypointsDatasetGrasp('train')
    # dataset_test = RobotKeypointsDatasetGrasp('test')

    # train_dataloader = DataLoader(dataset_train, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=False, drop_last=False)
    # test_dataloader = DataLoader(dataset_test, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False, drop_last=False)

    input_dim = 14
    output_dim = 7
    decoder_inp_dim = 7
    ## model
    # model = GeomVAE(
    #     input_dim,
    #     output_dim,
    #     FLAGS.latent_dimension,
    #     decoder_inp_dim,
    #     hidden_layers=[512, 512],
    # )

    # if FLAGS.pointnet:
    #     # model = JointPointVAE(
    #     #     input_dim,
    #     #     output_dim,
    #     #     FLAGS.latent_dimension,
    #     #     decoder_inp_dim,
    #     #     hidden_layers=[512, 512]
    #     # ).cuda()
    #     raise NotImplementedError
    # else:
    #     goal_model = GoalVAE(
    #         input_dim,
    #         output_dim,
    #         FLAGS.latent_dimension,
    #         decoder_inp_dim,
    #         hidden_layers=[256, 256]
    #     ).cuda()

    #     model = GeomVAE(
    #         input_dim,
    #         output_dim,
    #         FLAGS.latent_dimension,
    #         decoder_inp_dim,
    #         hidden_layers=[512, 512],
    #     ).cuda()
    # optimizer = Adam(model.parameters(), lr=FLAGS.lr, betas=(0.99, 0.999))


    # if FLAGS.resume_iter != 0:
        # model_path = osp.join(logdir, "model_{}".format(FLAGS.resume_iter))
        # model_path = '/home/anthony/repos/research/mpalm_affordances/training/gat/yilun_models/model_309000'
        # model_path = '/root/training/gat/yilun_models/model_309000'
        # model_path = '/root/training/gat/vae_cachedir/palm_poses_joint_hybrid_2/model_12000'
    if FLAGS.pointnet:
        # model_path = '/root/training/gat/vae_cachedir/pointnet_joint_2/model_30000'
        if FLAGS.primitive_name == 'grasp':
            goal_model_path = '/root/training/gat_vq/vae_cachedir/grasping_subgoal_indep_pointnet_0/model_40000'
            model_path = '/root/training/gat_vq/vae_cachedir/grasping_palms_indep_pointnet_0/model_40000'        
        elif FLAGS.primitive_name == 'pull':
            goal_model_path = '/root/training/gat_vq/vae_cachedir/pulling_subgoal_indep_pointnet_no_relu_0/model_25000'
            model_path = '/root/training/gat_vq/vae_cachedir/pulling_palms_indep_pointnet_no_relu_0/model_25000'        
        elif FLAGS.primitive_name == 'push':
            goal_model_path = '/root/training/gat_vq/vae_cachedir/pushing_subgoal_indep_pointnet_no_relu_0/model_25000'
            model_path = '/root/training/gat_vq/vae_cachedir/pushing_palms_indep_pointnet_no_relu_0/model_25000'                    

        goal_checkpoint = torch.load(goal_model_path)
        GOAL_FLAGS_OLD = goal_checkpoint['FLAGS']
        checkpoint = torch.load(model_path)
        FLAGS_OLD = checkpoint['FLAGS']            
        print('Loading from goal model path: ' + str(goal_model_path))
        print('Loading from model path: ' + str(model_path))

        model = PalmPointVAE(
            input_dim,
            output_dim,
            FLAGS_OLD.latent_dimension,
            decoder_inp_dim,
            hidden_layers=[256, 256]
        ).to(device)
        goal_model = GoalPointVAE(
            input_dim,
            output_dim,
            GOAL_FLAGS_OLD.latent_dimension,
            decoder_inp_dim,
            hidden_layers=[256, 256],
        ).to(device)              
    else:
        if FLAGS.primitive_name == 'grasp':
            # goal_model_path = '/root/training/gat_vq/vae_cachedir/grasping_h_subgoal_0/model_30000'
            # model_path = '/root/training/gat_vq/vae_cachedir/grasping_h_palms_both_0/model_30000'
            if FLAGS.H2:
                goal_model_path = '/root/training/gat_vq/vae_cachedir/grasping_H2_dense_subgoal_both_1/model_49000'
                model_path = '/root/training/gat_vq/vae_cachedir/grasping_h_palms_dense_0/model_26000'
            else:
                goal_model_path = '/root/training/gat_vq/vae_cachedir/grasping_h_subgoal_dense_vq_0/model_39000'
                # goal_model_path = '/root/training/gat_vq/vae_cachedir/grasping_h_subgoal_dense_0/model_29000'
                model_path = '/root/training/gat_vq/vae_cachedir/grasping_H_dense_palms_both_1/model_59000'
            if FLAGS.indep:
                # goal_model_path = '/root/training/gat_vq/vae_cachedir/grasping_h_subgoal_dense_vq_0/model_39000'
                goal_model_path = '/root/training/gat_vq/vae_cachedir/grasping_h_subgoal_dense_0/model_29000'                
                model_path = '/root/training/gat_vq/vae_cachedir/grasping_h_palms_dense_0/model_26000'               
            mask, transformation = True, True

            # model_path = '/root/training/gat_vq/vae_cachedir/grasping_h_palms_transformation_0'
            # mask, transformation = False, True

            # model_path = '/root/training/gat_vq/vae_cachedir/grasping_h_palms_mask_0'
            # mask, transformation = True, False                        
        elif FLAGS.primitive_name == 'pull':
            # goal_model_path = '/root/training/gat_vq/vae_cachedir/pulling_h_subgoal_0/model_30000'
            # model_path = '/root/training/gat_vq/vae_cachedir/pulling_h_palms_0/model_30000' 
            if FLAGS.H2:
                goal_model_path = '/root/training/gat_vq/vae_cachedir/pulling_H2_dense_subgoal_1/model_50000'
                model_path = '/root/training/gat_vq/vae_cachedir/pulling_h2_palms_dense_0/model_50000'                 
            else:
                goal_model_path = '/root/training/gat_vq/vae_cachedir/pulling_h_subgoal_dense_0/model_50000'
                model_path = '/root/training/gat_vq/vae_cachedir/pulling_H_dense_palms_1/model_50000'
            if FLAGS.indep:
                goal_model_path = '/root/training/gat_vq/vae_cachedir/pulling_h_subgoal_dense_0/model_50000'
                model_path = '/root/training/gat_vq/vae_cachedir/pulling_h2_palms_dense_0/model_50000'                   
            mask, transformation = False, True

        elif FLAGS.primitive_name == 'push':
            # goal_model_path = '/root/training/gat_vq/vae_cachedir/pushing_h_subgoal_0/model_49000'
            # model_path = '/root/training/gat_vq/vae_cachedir/pushing_h_palms_0/model_30000'   
            if FLAGS.H2:
                goal_model_path = '/root/training/gat_vq/vae_cachedir/pushing_H2_dense_subgoal_1/model_31000'
                model_path = '/root/training/gat_vq/vae_cachedir/pushing_h2_palms_dense_0/model_50000'                
            else:
                goal_model_path = '/root/training/gat_vq/vae_cachedir/pushing_h_subgoal_dense_0/model_50000'
                model_path = '/root/training/gat_vq/vae_cachedir/pushing_H_dense_palms_1/model_31000'
            if FLAGS.indep:
                goal_model_path = '/root/training/gat_vq/vae_cachedir/pushing_h_subgoal_dense_0/model_50000'
                model_path = '/root/training/gat_vq/vae_cachedir/pushing_h2_palms_dense_0/model_50000'                   
            mask, transformation = False, True
        else:
            raise ValueError('not recognized primitive')

        goal_checkpoint = torch.load(goal_model_path)
        GOAL_FLAGS_OLD = goal_checkpoint['FLAGS']
        checkpoint = torch.load(model_path)
        FLAGS_OLD = checkpoint['FLAGS']            
        print('Loading from goal model path: ' + str(goal_model_path))
        print('Loading from model path: ' + str(model_path))

        if FLAGS.indep:
            model = PalmVAE(
                input_dim,
                output_dim,
                FLAGS_OLD.latent_dimension,
                decoder_inp_dim,
                hidden_layers=[256, 256]
            ).cuda()            
            goal_model = GoalVAE(
                input_dim,
                output_dim,
                GOAL_FLAGS_OLD.latent_dimension,
                decoder_inp_dim,
                hidden_layers=[256, 256]
            ).cuda()
        else:
            if FLAGS.H2:
                model = PalmVAE(
                    input_dim,
                    output_dim,
                    FLAGS_OLD.latent_dimension,
                    decoder_inp_dim,
                    hidden_layers=[256, 256]
                ).cuda()

                goal_model = GeomGoalVAE(
                    input_dim,
                    output_dim,
                    GOAL_FLAGS_OLD.latent_dimension,
                    decoder_inp_dim,
                    hidden_layers=[512, 512],
                    mask=mask,
                    transformation=transformation
                ).cuda()   

            else:
                goal_model = GoalVAE(
                    input_dim,
                    output_dim,
                    GOAL_FLAGS_OLD.latent_dimension,
                    decoder_inp_dim,
                    hidden_layers=[256, 256]
                ).cuda()

                model = GeomPalmVAE(
                    input_dim,
                    output_dim,
                    FLAGS_OLD.latent_dimension,
                    decoder_inp_dim,
                    hidden_layers=[512, 512],
                    mask=mask,
                    transformation=transformation
                ).cuda()                

    # try:
    goal_model.load_state_dict(goal_checkpoint['model_state_dict'])
    goal_eval = ModelEvaluator(goal_model, GOAL_FLAGS_OLD)

    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    contact_eval = ModelEvaluator(model, FLAGS_OLD)

    signal.signal(signal.SIGINT, signal_handler)
    model = model.eval()
    while True:
        observation_avail = len(os.listdir(FLAGS.observation_dir)) > 0
        if observation_avail:
            # for fname in os.listdir(FLAGS.observation_dir):
            for fname in os.listdir(FLAGS.observation_dir):
                valid = False
                if FLAGS.primitive_name == 'pull' and fname.startswith('pull'):
                    valid = True
                elif FLAGS.primitive_name == 'push' and fname.startswith('push'):
                    valid = True
                elif FLAGS.primitive_name == 'grasp' and fname.startswith('grasp'):
                    valid = True            
                if fname.endswith('.npz') and valid:
                    time.sleep(0.5)
                    if FLAGS.indep:
                        test_indep(goal_eval, contact_eval, model, goal_model, fname, FLAGS, [model_path, goal_model_path])
                    else:
                        if FLAGS.H2:
                            test_joint2(goal_eval, contact_eval, model, goal_model, fname, FLAGS, [model_path, goal_model_path])                        
                        else:
                            test_joint(goal_eval, contact_eval, model, goal_model, fname, FLAGS, [model_path, goal_model_path])
        time.sleep(0.01)


def main():
    FLAGS = parser.parse_args()
    print(FLAGS)


    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    else:
        main_single(0, FLAGS)


if __name__ == "__main__":
    main()