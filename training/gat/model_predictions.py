import torch
import torch.nn.functional as F
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from models_vae import VAE, GeomVAE, JointVAE, JointPointVAE
import sys, signal
import time


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
# parser.add_argument('--prediction_dir', type=str, default='/root/catkin_ws/src/primitives/predictions')
# parser.add_argument('--observation_dir', type=str, default='/root/catkin_ws/src/primitives/observations')
parser.add_argument('--prediction_dir', type=str, default='/tmp/predictions')
parser.add_argument('--observation_dir', type=str, default='/tmp/observations')
parser.add_argument('--pointnet', action='store_true')
parser.add_argument('--pulling', action='store_true')


def test(model, obs_file, FLAGS):
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
        start = observation['pointcloud_pts']
        transformation = observation['transformation']
        kd_idx = np.arange(100, dtype=np.int64)

        start = torch.from_numpy(start)
        transformation = torch.from_numpy(transformation)
        kd_idx = torch.from_numpy(kd_idx)

        joint_keypoint = torch.cat([start, transformation[None, :].repeat(100, 1)], dim=1)
        joint_keypoint = joint_keypoint[None, :, :]
        kd_idx = kd_idx[None, :]

        joint_keypoint = joint_keypoint.float().to(dev)
        kd_idx = kd_idx.long().to(dev)

        z = torch.randn(1, FLAGS.latent_dimension).to(dev)

        geom_embed = model.geom_encoder(joint_keypoint)
        repeat = kd_idx.size(1)
        ex_wt = model.exist_wt(geom_embed)
        kd_idx = kd_idx[:, :, None].repeat(1, 1, geom_embed.size(2))[:, :repeat]
        geom_embed = torch.gather(geom_embed, 1, kd_idx)

        size = geom_embed.size()
        z = z[:, None, :].repeat(1, size[1], 1)
        z = (z * geom_embed)

        inp = torch.cat((z, geom_embed), dim=-1)
        recon_mu = model.decoder(inp)
        output_r, output_l = recon_mu

        output_r, output_l = output_r.detach().cpu().numpy(), output_l.detach().cpu().numpy()
        output_joint = np.concatenate([output_r, output_l], axis=2)
        ex_wt = ex_wt.detach().cpu().numpy().squeeze()
        sort_idx = np.argsort(ex_wt)[None, :]
        predictions = []

        for i in range(output_joint.shape[0]):
            for j in range(output_joint.shape[1]):
                j = sort_idx[i, j]
                pred_info = output_joint[i, j]
                predictions.append(pred_info.tolist())
        predictions = np.asarray(predictions).squeeze()
        output_file = osp.join(FLAGS.prediction_dir, obs_file)
        np.savez(output_file, prediction=predictions)
        os.remove(osp.join(FLAGS.observation_dir, obs_file))


def test_joint(model, obs_file, FLAGS):
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
        start = observation['pointcloud_pts']
        # transformation = observation['transformation']
        if not FLAGS.pulling:
            start_mean = np.mean(start, axis=0, keepdims=True)
        else:
            start_mean = np.array([0.0, 0.0, 0.0])
        # start_mean = np.mean(start, axis=0, keepdims=True)
        start_normalized = (start - start_mean)
        start_mean = np.tile(start_mean, (start.shape[0], 1))

        start = np.concatenate([start_normalized, start_mean], axis=1)
        kd_idx = np.arange(100, dtype=np.int64)

        start = torch.from_numpy(start)
        kd_idx = torch.from_numpy(kd_idx)

        joint_keypoint = start
        joint_keypoint = joint_keypoint[None, :, :]
        kd_idx = kd_idx[None, :]

        joint_keypoint = joint_keypoint.float().to(dev)
        kd_idx = kd_idx.long().to(dev)

        palm_predictions = []
        mask_predictions = []
        trans_predictions = []

        for repeat in range(10):
            palm_repeat = []
            z = torch.randn(1, FLAGS.latent_dimension).to(dev)
            recon_mu, ex_wt = model.decode(z, joint_keypoint)
            output_r, output_l, pred_mask, pred_trans = recon_mu
            # output_r, output_l, pred_trans = recon_mu
            # pred_mask = torch.zeros_like(start)
            mask_predictions.append(pred_mask.detach().cpu().numpy())
            trans_predictions.append(pred_trans.detach().cpu().numpy())

            output_r, output_l = output_r.detach().cpu().numpy(), output_l.detach().cpu().numpy()

            if FLAGS.pointnet:
                output_joint = np.concatenate([output_r, output_l], axis=1)
                palm_repeat.append(output_joint)
            else:
                output_joint = np.concatenate([output_r, output_l], axis=2)
                ex_wt = ex_wt.detach().cpu().numpy().squeeze()
                # sort_idx = np.argsort(ex_wt, axis=1)[:, ::-1]
                sort_idx = np.argsort(ex_wt)[None, :]

                for i in range(output_joint.shape[0]):
                    for j in range(output_joint.shape[1]):
                        j = sort_idx[i, j]
                        pred_info = output_joint[i, j]
                #         pred_info = obj_frame[i].cpu().numpy()
                        palm_repeat.append(pred_info.tolist())
            palm_predictions.append(palm_repeat)
        palm_predictions = np.asarray(palm_predictions).squeeze()
        mask_predictions = np.asarray(mask_predictions).squeeze()
        trans_predictions = np.asarray(trans_predictions).squeeze()

        # predictions = np.asarray(predictions).squeeze()
        output_file = osp.join(FLAGS.prediction_dir, obs_file)
        # print('Saving to: ' + str(output_file))
        np.savez(
            output_file,
            palm_predictions=palm_predictions,
            mask_predictions=mask_predictions,
            trans_predictions=trans_predictions)
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

    if FLAGS.pointnet:
        model = JointPointVAE(
            input_dim,
            output_dim,
            FLAGS.latent_dimension,
            decoder_inp_dim,
            hidden_layers=[512, 512]
        ).cuda()
    else:
        model = JointVAE(
            input_dim,
            output_dim,
            FLAGS.latent_dimension,
            decoder_inp_dim,
            hidden_layers=[512, 512],
        ).cuda()
    optimizer = Adam(model.parameters(), lr=FLAGS.lr, betas=(0.99, 0.999))


    # if FLAGS.resume_iter != 0:
        # model_path = osp.join(logdir, "model_{}".format(FLAGS.resume_iter))
        # model_path = '/home/anthony/repos/research/mpalm_affordances/training/gat/yilun_models/model_309000'
        # model_path = '/root/training/gat/yilun_models/model_309000'
        # model_path = '/root/training/gat/vae_cachedir/palm_poses_joint_hybrid_2/model_12000'
    if FLAGS.pulling:
        # model_path = '/root/training/gat/vae_cachedir/pulling_joint_gat_1/model_9000'
        # model_path = '/root/training/gat/vae_cachedir/pulling_joint_gat_0/model_3000'   
        # model_path = '/root/training/gat/vae_cachedir/pulling_joint_gat_trans_0/model_10000'
        model_path = '/root/training/gat/vae_cachedir/pulling_joint_gat_trans_unnormalized_0/model_10000'      
    else:
        if FLAGS.pointnet:
            model_path = '/root/training/gat/vae_cachedir/pointnet_joint_2/model_30000'
        else:
            model_path = '/root/training/gat/vae_cachedir/joint_hybrid_poses_diverse_0/model_20000'
            # model_path = '/root/training/gat/vae_cachedir/joint_gat_transformation_subgoal_0/model_25000'            
    print('Loading from model path: ' + str(model_path))
    checkpoint = torch.load(model_path)
    FLAGS_OLD = checkpoint['FLAGS']

    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        model_state_dict = {k.replace("module.", "") : v for k, v in checkpoint['model_state_dict'].items()}


    if FLAGS.gpus > 1:
        sync_model(model)

    if not osp.exists(FLAGS.prediction_dir):
        os.makedirs(FLAGS.prediction_dir)
    if not osp.exists(FLAGS.observation_dir):
        os.makedirs(FLAGS.observation_dir)

    signal.signal(signal.SIGINT, signal_handler)
    model = model.eval()
    while True:
        observation_avail = len(os.listdir(FLAGS.observation_dir)) > 0
        if observation_avail:
            for fname in os.listdir(FLAGS.observation_dir):
                valid = False
                if FLAGS.pulling and fname.startswith('pull'):
                    valid = True
                elif not FLAGS.pulling and fname.startswith('grasp'):
                    valid = True
                if fname.endswith('.npz') and valid:
                    time.sleep(0.5)
                    test_joint(model, fname, FLAGS)
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