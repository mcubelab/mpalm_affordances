import torch
import torch.nn.functional as F
import os
import torch.multiprocessing as mp
import numpy as np
import os.path as osp
from torch.utils.data import DataLoader
from easydict import EasyDict
from torch.nn.utils import clip_grad_norm
import pdb
import torch.nn as nn

from data import RobotKeypointsDatasetGraspMask
from random import choices
# from apex.optimizers import FusedAdam
from torch.optim import Adam
import argparse
from itertools import permutations
import itertools
import matplotlib.pyplot as plt
from models_vae import VAE, GeomVAE, GoalVAE
from vis_utils import plot_pointcloud
# from baselines.logger import TensorBoardOutputFormat
import plotly



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
                    default=128)
parser.add_argument('--kl_anneal_rate', type=float, default=0.9999)


def average_gradients(model):
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def sync_model(model):
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.broadcast(param.data, 0)


def test(test_dataloader, model, FLAGS):
    model = model.eval()
    vae = model
    counter = 0

    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    vae = vae.to(dev)

    with torch.no_grad():
        for start, object_mask_down, translation in test_dataloader:
            object_mask_down = object_mask_down[:, :, None].float()
            start_cpu = start = start.float()
            joint_keypoint = torch.cat([start, object_mask_down], dim=2)

            joint_keypoint = joint_keypoint.float().to(dev)
            start = start.to(dev)
            object_mask_down = object_mask_down.float().to(dev)

            # Sample 10 different keypoints from the model
            for repeat in range(10):
                z = torch.randn(start.size(0), FLAGS.latent_dimension).to(dev)
                pred_mask, trans, _ = model.decode(z, start)
                pred_mask = pred_mask.detach().cpu().numpy()

                for i in range(pred_mask.shape[0]):
                    output_file = osp.join("output_vae_goal", "mask_{}_repeat_{}.npz".format(i, repeat))
                    np.savez(output_file, start=start_cpu.numpy()[i], pred_mask=pred_mask[i].squeeze())
                    # fig = plot_pointcloud(start_cpu.numpy()[i], pred_mask[i].squeeze())
                    # plotly.io.write_image(fig, output_file)

            assert False


def train(train_dataloader, test_dataloader, model, optimizer, FLAGS, logdir, rank_idx):
    it = int(FLAGS.resume_iter)
    optimizer.zero_grad()

    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    model = model.to(dev).train()
    vae = model
    kl_weight = 1.0

    ex_criterion = nn.BCELoss()

    for epoch in range(FLAGS.num_epoch):
        kl_weight = FLAGS.kl_anneal_rate*kl_weight
        kl_coeff = 1 - kl_weight

        for start, object_mask_down, translation in train_dataloader:
            object_mask_down = object_mask_down[:, :, None].float()
            start = start.float()
            translation = translation.float()
            translation_epd = translation[:, None, :].repeat(1, start.size(1), 1)
            joint_keypoint = torch.cat([start, object_mask_down, translation_epd], dim=2)

            joint_keypoint = joint_keypoint.float().to(dev)
            start = start.to(dev)
            object_mask_down = object_mask_down.float().to(dev)
            translation = translation.to(dev)

            z, recon_mu, z_mu, z_logvar = vae.forward(joint_keypoint, start)
            kl_loss = vae.kl_loss(z_mu, z_logvar)

            exist_prob, pred_trans, loss_vq = recon_mu

            existence_loss = vae.existence_loss(exist_prob, object_mask_down)
            trans_loss = vae.translation_loss(pred_trans, translation)

            recon_loss = existence_loss + trans_loss
            loss = kl_coeff*kl_loss + recon_loss + loss_vq
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            it += 1


            if it % FLAGS.log_interval == 0 and rank_idx == 0:

                kvs = {}
                kvs['kl_loss'] = kl_loss.item()
                kvs['recon_loss'] = recon_loss.item()
                kvs['trans_loss'] = trans_loss.item()
                kvs['vq_loss'] = loss_vq.item()
                kvs['existence_loss'] = existence_loss.item()
                # kvs['grad_mag'] = torch.abs(contact_obj_grad).mean().item()
                kvs['loss'] = loss.item()
                string = "Iteration {} with values of ".format(it)

                for k, v in kvs.items():
                    string += "%s: %.3f, " % (k,v)
                    # string += "{}: {.3f}, ".format(k, v)
                    # logger.writekvs(kvs)

                if FLAGS.gpus > 1:
                    average_gradients(model)

                print(string)

            if it % FLAGS.save_interval == 0 and rank_idx == 0:
                # model = model.eval()
                model_path = osp.join(logdir, "model_{}".format(it))
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'FLAGS': FLAGS}, model_path)
                print("Saving model in directory....")


                ## running test
                print('run test')
                # test(test_dataloader, model, rotation_criterion, FLAGS, logdir, rank_idx, step=it)
                # model = model.train()


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
    dataset_train = RobotKeypointsDatasetGraspMask('train')
    dataset_test = RobotKeypointsDatasetGraspMask('test')

    train_dataloader = DataLoader(dataset_train, num_workers=0, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=False, drop_last=False)
    test_dataloader = DataLoader(dataset_test, num_workers=0, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False, drop_last=False)

    input_dim = 14
    output_dim = 7
    decoder_inp_dim = 7
    ## model
    model = GoalVAE(
        input_dim,
        output_dim,
        FLAGS.latent_dimension,
        decoder_inp_dim,
        hidden_layers=[256, 256],
    ).to(device)
    optimizer = Adam(model.parameters(), lr=FLAGS.lr, betas=(0.99, 0.999))


    if FLAGS.resume_iter != 0:
        model_path = osp.join(logdir, "model_{}".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path)
        FLAGS_OLD = checkpoint['FLAGS']

        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model_state_dict = {k.replace("module.", "") : v for k, v in checkpoint['model_state_dict'].items()}


    if FLAGS.gpus > 1:
        sync_model(model)


    if FLAGS.train:
        model = model.train()
        train(train_dataloader, test_dataloader, model, optimizer, FLAGS, logdir, rank_idx)
    else:
        model = model.eval()
        test(test_dataloader, model, FLAGS)


def main():
    FLAGS = parser.parse_args()
    print(FLAGS)


    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    else:
        main_single(0, FLAGS)


if __name__ == "__main__":
    main()
