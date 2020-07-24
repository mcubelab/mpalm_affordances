import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import os.path as osp
from torch.utils.data import DataLoader
from easydict import EasyDict
from torch.nn.utils import clip_grad_norm
import pdb
import torch.nn as nn

from data import RobotKeypointsDatasetGrasp, RobotKeypointsDatasetGraspJoint
from random import choices
# from apex.optimizers import FusedAdam
from torch.optim import Adam
import argparse
from itertools import permutations
import itertools
import matplotlib.pyplot as plt
# from models_vae import VAE, GeomVAE
from models_vae_h import GeomPalmVAE, GeomGoalVAE
from datetime import datetime
# from baselines.logger import TensorBoardOutputFormat



"""Parse input arguments"""
parser = argparse.ArgumentParser(description='Train reasoning model')
parser.add_argument('--train', action='store_true', help='whether or not to train')
parser.add_argument('--table_mesh', action='store_true', help='whether or not to add a mesh to the table')
parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not')

# Generic Parameters for Experiments
parser.add_argument('--dataset', default='intphys', type=str, help='intphys or others')
parser.add_argument('--logdir', default='vae_cachedir', type=str, help='location where log of experiments will be stored')
parser.add_argument('--exp', default='debug', type=str, help='name of experiments')
parser.add_argument('--data_workers', default=4, type=int, help='Number of different data workers to load data in parallel')

# train
parser.add_argument('--batch_size', default=128, type=int, help='size of batch of input to use')
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

parser.add_argument('--pulling', action='store_true')
parser.add_argument('--subgoal', type=str, default='mask')
parser.add_argument('--notes', type=str, default="")
parser.add_argument('--palms', action='store_true')


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
        for start, transformation, obj_frame, kd_idx, vis_info, decoder_x, obj_id in test_dataloader:
            # joint_keypoint = torch.cat([start, transformation[:, None, :].repeat(1, start.size(1), 1)], dim=2)
            joint_keypoint = torch.cat([start.float(), transformation.float()], dim=2)

            joint_keypoint = joint_keypoint.float().to(dev)
            obj_frame = obj_frame.float().to(dev)
            kd_idx = kd_idx.long().to(dev)
            decoder_x = decoder_x.float().to(dev)

            z = torch.randn(obj_frame.size(0), FLAGS.latent_dimension).to(dev)
            recon_mu, ex_wt = model.decode(z, joint_keypoint, kd_idx, full=True)
            output_r, output_l = recon_mu

            output_r, output_l = output_r.detach().cpu().numpy(), output_l.detach().cpu().numpy()
            output_joint = np.concatenate([output_r, output_l], axis=2)
            ex_wt = ex_wt.detach().cpu().numpy().squeeze()
            sort_idx = np.argsort(ex_wt, axis=1)[:, ::-1]

            for i in range(output_joint.shape[0]):
                for j in range(output_joint.shape[1]):
                    output_file = osp.join("output_vae", "state_{}_{}.npz".format(counter, j))
                    j = sort_idx[i, j]
                    vis_info_i = vis_info[i]
                    pred_info = output_joint[i, j]
                    # pred_info = obj_frame[i].cpu().numpy()
                    total_info = np.concatenate([vis_info_i, pred_info])
                    np.savez(output_file, data=total_info, id=obj_id[i])

                counter += 1

                if counter > 15:
                    break


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

                
        # for start, object_mask_down, transformation, obj_frame, kd_idx, decoder_x in train_dataloader:
        for start, transformation, obj_frame, kd_idx, decoder_x, object_mask_down, translation in train_dataloader:
            # joint_keypoint = torch.cat([start, transformation[:, None, :].repeat(1, start.size(1), 1)], dim=2)
            start = start.float()
            object_mask_down = object_mask_down[:, :, None].float()

            # joint_keypoint = torch.cat(
            #     [start, 
            #     object_mask_down,
            #     obj_frame[:, None, :].repeat(1, start.size(1), 1).float(), 
            #     transformation[:, None, :].repeat(1, start.size(1), 1).float()], dim=2)
            
            start = start.to(dev)
            object_mask_down = object_mask_down.float().to(dev)
            obj_frame = obj_frame.float().to(dev)
            kd_idx = kd_idx.long().to(dev)
            decoder_x = decoder_x.float().to(dev)
            transformation = transformation.float().to(dev)                        
            
            if FLAGS.palms:
                if FLAGS.subgoal == 'both':
                    decoder_x = torch.cat([
                        start, 
                        object_mask_down,
                        transformation[:, None, :].repeat(1, start.size(1), 1)], dim=2)
                    joint_keypoint = torch.cat(
                        [start, 
                        object_mask_down,
                        obj_frame[:, None, :].repeat(1, start.size(1), 1).float(), 
                        transformation[:, None, :].repeat(1, start.size(1), 1).float()], dim=2)                        
                
                elif FLAGS.subgoal == 'mask':
                    decoder_x = torch.cat([
                        start, 
                        object_mask_down], dim=2)  
                    joint_keypoint = torch.cat(
                        [start, 
                        object_mask_down,
                        obj_frame[:, None, :].repeat(1, start.size(1), 1).float()], dim=2)                                  

                elif FLAGS.subgoal == 'transformation':
                    decoder_x = torch.cat([
                        start, 
                        transformation[:, None, :].repeat(1, start.size(1), 1)], dim=2)
                    joint_keypoint = torch.cat(
                        [start, 
                        obj_frame[:, None, :].repeat(1, start.size(1), 1).float(), 
                        transformation[:, None, :].repeat(1, start.size(1), 1).float()], dim=2)                        
                else:
                    raise ValueError('subgoal rep not recognized')
                joint_keypoint = joint_keypoint.float().to(dev)

                z, recon_mu, z_mu, z_logvar = vae.forward(joint_keypoint, decoder_x, kd_idx)
                kl_loss = vae.kl_loss(z_mu, z_logvar)
                output_r, output_l, ex_wt = recon_mu

                obj_frame = obj_frame[:, None, :].repeat(1, output_r.size(1), 1)
                s = obj_frame.size()
                obj_frame = obj_frame.view(s[0]*s[1], s[2])
                output_r = output_r.view(s[0]*s[1], s[2] // 2)
                output_l = output_l.view(s[0]*s[1], s[2] // 2)

                target_batch_left, target_batch_right = torch.chunk(obj_frame, 2, dim=1)

                pos_loss_right = vae.mse(
                    output_r[:, :3],
                    target_batch_right[:, :3])
                ori_loss_right = vae.rotation_loss(
                    output_r[:, 3:],
                    target_batch_right[:, 3:]
                )

                pos_loss_left = vae.mse(
                    output_l[:, :3],
                    target_batch_left[:, :3])
                ori_loss_left = vae.rotation_loss(
                    output_l[:, 3:],
                    target_batch_left[:, 3:]
                )

                pos_loss = pos_loss_left + pos_loss_right
                # ori_loss = pos_2_loss_left + pos_2_loss_right
                ori_loss = ori_loss_left + ori_loss_right

                label = torch.zeros_like(ex_wt)
                label = label.scatter(1, kd_idx[:, :20, None], 1)
                exist_loss = ex_criterion(ex_wt, label)

                recon_loss = pos_loss + ori_loss + exist_loss
            else:
                
                joint_keypoint = torch.cat(
                    [start, 
                    object_mask_down,
                    obj_frame[:, None, :].repeat(1, start.size(1), 1).float(), 
                    transformation[:, None, :].repeat(1, start.size(1), 1).float()], dim=2)
                joint_keypoint = joint_keypoint.float().to(dev)
                
                decoder_x = torch.cat([
                    start, 
                    obj_frame[:, None, :].repeat(1, start.size(1), 1)], dim=2) 

                z, recon_mu, z_mu, z_logvar = vae.forward(joint_keypoint, decoder_x, kd_idx)

                pred_mask, _, pred_trans = recon_mu

                kl_loss = vae.kl_loss(z_mu, z_logvar)

                mask_loss = vae.existence_loss(pred_mask, object_mask_down)

                trans_loss_pos = vae.translation_loss(pred_trans[:, :3], transformation[:, :3])
                trans_loss_ori = vae.rotation_loss(pred_trans[:, 3:], transformation[:, 3:])
                trans_loss = trans_loss_pos + trans_loss_ori

                recon_loss = mask_loss + trans_loss    

            loss = kl_coeff*kl_loss + recon_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            it += 1


            if it % FLAGS.log_interval == 0 and rank_idx == 0:

                kvs = {}
                kvs['kl_loss'] = kl_loss.item()
                kvs['recon_loss'] = recon_loss.item()
                kvs['loss'] = loss.item()
                if FLAGS.palms:
                    kvs['pos_loss'] = pos_loss.item()
                    kvs['ori_loss'] = ori_loss.item()
                else:
                    kvs['trans_loss'] = trans_loss.item()
                    kvs['mask_loss'] = mask_loss.item()                    
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
        now = datetime.now()
        nowstr = now.strftime("_%m_%d_%y_%H-%M-%S")
        notes = ""
        notes += nowstr + '\n'
        notes += FLAGS.notes 
        notes += '\n'
        with open(osp.join(logdir, 'exp_notes.txt'), 'w') as f:
            f.write(notes)
    # logger = TensorBoardOutputFormat(logdir)

    ## dataset
    dataset_train = RobotKeypointsDatasetGraspJoint('train', pulling=FLAGS.pulling)
    dataset_test = RobotKeypointsDatasetGraspJoint('test', pulling=FLAGS.pulling)

    train_dataloader = DataLoader(dataset_train, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=False, drop_last=False)
    test_dataloader = DataLoader(dataset_test, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False, drop_last=False)

    input_dim = 14
    output_dim = 7
    decoder_inp_dim = 7

    mask = True
    transformation = False
    if FLAGS.pulling:
        mask = False
        transformation = True
    else:
        if FLAGS.subgoal == 'mask':
            mask = True
            transformation = False
        elif FLAGS.subgoal == 'transformation':
            mask = False
            transformation = True 
        elif FLAGS.subgoal == 'both':
            mask = True
            transformation = True
        else:
            raise ValueError('subgoal not recognized')
    ## model
    # model = GeomVAE(
    #     input_dim,
    #     output_dim,
    #     FLAGS.latent_dimension,
    #     decoder_inp_dim,
    #     hidden_layers=[512, 512],
    #     table_mesh=FLAGS.table_mesh,
    #     mask=mask,
    #     transformation=transformation
    # ).cuda()
    if FLAGS.palms:
        model = GeomPalmVAE(
            input_dim,
            output_dim,
            FLAGS.latent_dimension,
            decoder_inp_dim,
            hidden_layers=[512, 512],
            table_mesh=FLAGS.table_mesh,
            mask=mask,
            transformation=transformation
        ).cuda()
    else:
        model = GeomGoalVAE(
            input_dim,
            output_dim,
            FLAGS.latent_dimension,
            decoder_inp_dim,
            hidden_layers=[512, 512],
            table_mesh=FLAGS.table_mesh,
            mask=mask,
            transformation=transformation
        ).cuda()                

    optimizer = Adam(model.parameters(), lr=FLAGS.lr, betas=(0.9, 0.999))

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
