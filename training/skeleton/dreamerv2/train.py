import os, os.path as osp
import sys
import argparse
import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical, kl_divergence

from models import TransitionModelSeqBatch
from dreamer_utils import SkillLanguage, prepare_sequence_tokens, process_pointcloud_sequence_batch, process_pointcloud_batch
from data import SkeletonTransitionDataset

PAD_token = 0
SOS_token = 1
EOS_token = 2


def eval(dataloader, model, language, logdir, args, tmesh=False):
    model = model.eval()

    if args.cuda:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    with torch.no_grad():
        for sample in dataloader:
            observations, action_tokens, tables, mask, reward, transformation, goal = sample

            # subtract and concatenate mean to point cloud features
            observations = observations.float().to(dev)
            observations = process_pointcloud_sequence_batch(observations)
            tables = tables.float().to(dev)
            mask_gt = mask.float().to(dev)
            mask_gt = torch.cat((mask_gt, mask_gt[:, -1][:, None, :, :]), dim=1)
            reward = reward.float().to(dev)
            transformation = transformation.float().to(dev)
            goal = goal.float().to(dev)
            goal = process_pointcloud_batch(goal)

            # convert action sequences to one-hot
            s = action_tokens.size()
            action_tokens = action_tokens.long().to(dev)
            action_tokens_oh = torch.FloatTensor(s[0], s[1], len(language.skill2index)).to(dev)
            action_tokens_oh.zero_()
            action_tokens_oh.scatter_(2, action_tokens[:, :, None], 1)

            # initialize GRU hidden state
            batch_size, h_size = observations.size(0), model.hidden_init.size(-1)
            init_h = torch.zeros(batch_size, 1, h_size).to(dev)

            # get sequence of latent states and contact-mask reconstructions
            z_post, z_post_logits, z_prior, z_prior_logits, x_mask, h, reward_pred = model.forward_loop(
                observations, 
                action_tokens_oh, 
                init_h,
                mask_gt,
                transformation,
                goal)

            # let's look at the points
            for b_idx in range(batch_size):
                pcds = observations[b_idx]         
                for i in range(pcds.size(0)):
                    pcd = pcds[i].cpu().numpy().squeeze()[:, :3] + pcds[i].cpu().numpy().squeeze()[0, 3:]

                    mask = x_mask[b_idx, i].detach().cpu().numpy()
                    gt_mask = mask_gt[b_idx, i].detach().cpu().numpy()
                    top_inds = np.argsort(mask)[::-1]
                    # print('top inds: ', top_inds)
                    print(mask)
                    pred_mask = np.zeros((mask.shape[0]), dtype=bool)
                    pred_mask[top_inds[:15]] = True

                    if tmesh:
                        import trimesh
                        ttable = trimesh.PointCloud(tables[b_idx].detach().cpu().data.numpy().squeeze())
                        ttable.colors = np.tile(np.array([255, 0, 0, 255]), (ttable.vertices.shape[0], 1))   

                        tpcd = trimesh.PointCloud(pcd)
                        tpcd.colors = np.tile(np.array([255, 0, 255, 255]), (tpcd.vertices.shape[0], 1))

                        tpcd.colors[np.where(pred_mask)[0]] = np.tile(np.array([0, 0, 255, 255]), (np.where(pred_mask)[0].shape[0], 1))  
                        # tpcd.colors[np.where(gt_mask)[0]] = np.tile(np.array([0, 0, 255, 255]), (np.where(gt_mask)[0].shape[0], 1))                          

                        scene = trimesh.Scene()
                        scene.add_geometry([tpcd, ttable])
                        # scene.show()            


def train(dataloader, model, optimizer, language, logdir, args):
    model = model.train()

    if args.cuda:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')
    
    it = 0
    for epoch in range(1, args.num_epoch):
        for sample in dataloader:
            it += 1
            observations, action_tokens, tables, mask, reward, transformation, goal = sample

            # subtract and concatenate mean to point cloud features
            observations = observations.float().to(dev)
            observations = process_pointcloud_sequence_batch(observations)
            tables = tables.float().to(dev)
            mask_gt = mask.float().to(dev)
            mask_gt = torch.cat((mask_gt, mask_gt[:, -1][:, None, :, :]), dim=1)
            reward = reward.float().to(dev)
            transformation = transformation.float().to(dev)
            goal = goal.float().to(dev)
            goal = process_pointcloud_batch(goal)

            # convert action sequences to one-hot
            s = action_tokens.size()
            action_tokens = action_tokens.long().to(dev)
            action_tokens_oh = torch.FloatTensor(s[0], s[1], len(language.skill2index)).to(dev)
            action_tokens_oh.zero_()
            action_tokens_oh.scatter_(2, action_tokens[:, :, None], 1)

            # initialize GRU hidden state
            batch_size, h_size = observations.size(0), model.hidden_init.size(-1)
            init_h = torch.randn(batch_size, 1, h_size).to(dev)

            # get sequence of latent states and contact-mask reconstructions
            z_post, z_post_logits, z_prior, z_prior_logits, x_mask, h, reward_pred = model.forward_loop(
                observations, 
                action_tokens_oh, 
                init_h,
                mask_gt,
                transformation,
                goal)

            # compute KL losses
            p, p_sg = Categorical(logits=z_prior_logits), Categorical(logits=z_prior_logits.detach())
            q, q_sg = Categorical(logits=z_post_logits), Categorical(logits=z_post_logits.detach())
            kld_1 = kl_divergence(p, q_sg)
            kld_2 = kl_divergence(p_sg, q)

            kl_loss = kld_1.mean() + kld_2.mean()

            # compute reconstruction loss
            mask_loss = model.bce(x_mask, mask_gt)

            # from IPython import embed
            # embed()
            # predict rewards/probability of reaching the goal -- TODO
            reward_loss = model.mse(reward_pred, reward)

            # predict reward losses -- TODO

            optimizer.zero_grad()
            loss = mask_loss + reward_loss
            loss.backward()
            optimizer.step()

            if it % args.log_interval == 0:

                kvs = {}
                kvs['epoch'] = epoch
                kvs['kl_loss'] = kl_loss.item()
                kvs['mask_loss'] = mask_loss.item()
                kvs['reward_loss'] = reward_loss.item()
                kvs['loss'] = loss.item()
                string = "Iteration {} with values of ".format(it)

                for k, v in kvs.items():
                    string += "%s: %.5f, " % (k,v)

                print(string)

            if it % args.save_interval == 0:
                # model = model.eval()
                model_path = osp.join(logdir, "model_{}".format(it))
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'FLAGS': args}, model_path)
                print("Evaluating...")    
                eval(dataloader, model, language, logdir, args)
                model = model.train()



def main(args):
    
    dataset = SkeletonTransitionDataset()
    train_loader = DataLoader(dataset, batch_size=2)

    if torch.cuda.is_available() and args.cuda:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')   
    
    skill_lang = dataset.skill_lang
    transition_model = TransitionModelSeqBatch(
        observation_dim=6,
        action_dim=len(skill_lang.skill2index),
        latent_dim=256,
        out_dim=256,
        cat_dim=32).cuda() 

    optimizer = torch.optim.Adam(transition_model.parameters(), lr=args.lr)


    logdir = osp.join(args.logdir, args.exp)
    if not osp.exists(logdir):
        os.makedirs(logdir)

    train(train_loader, transition_model, optimizer, skill_lang, logdir, args)   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='whether or not to train')
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not')

    # Generic Parameters for Experiments
    parser.add_argument('--dataset', default='intphys', type=str, help='intphys or others')
    parser.add_argument('--logdir', default='cachedir', type=str, help='location where log of experiments will be stored')
    parser.add_argument('--exp', default='debug', type=str, help='name of experiments')
    parser.add_argument('--data_workers', default=4, type=int, help='Number of different data workers to load data in parallel')

    # train
    parser.add_argument('--batch_size', default=128, type=int, help='size of batch of input to use')
    parser.add_argument('--num_epoch', default=10000, type=int, help='number of epochs of training to run')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for training')
    parser.add_argument('--log_interval', default=10, type=int, help='log outputs every so many batches')
    parser.add_argument('--save_interval', default=1000, type=int, help='save outputs every so many batches')
    parser.add_argument('--resume_iter', default=0, type=str, help='iteration to resume training')

    # model 
    parser.add_argument('--latent_dim', default=512, type=int, help='size of hidden representation')
    parser.add_argument('--max_seq_length', default=5, type=int, help='maximum sequence length')

    args = parser.parse_args()

    main(args)