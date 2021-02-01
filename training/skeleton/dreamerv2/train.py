import os, os.path as osp
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from io import BytesIO
import trimesh

from models import TransitionModelSeqBatch
from dreamer_utils import SkillLanguage, prepare_sequence_tokens, process_pointcloud_sequence_batch, process_pointcloud_batch
from data import SkeletonTransitionDataset, SkeletonDataset

PAD_token = 0
SOS_token = 1
EOS_token = 2


def eval_reward(dataloader, model, language, logdir, writer, args, it, tmesh=False):
    model = model.eval()

    if args.cuda:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    with torch.no_grad():
        k = 0
        global_step = 0
        for sample in dataloader:
            k += 1
            # from IPython import embed
            # embed()
            observations, action_str, tables, mask, reward_step, transformation, goal = sample 

            # subtract and concatenate mean to point cloud features
            raw_observations = observations.float().to(dev)
            observations = process_pointcloud_batch(raw_observations)
            tables = tables.float().to(dev)
            transformation = transformation.float().to(dev)
            goal = goal.float().to(dev)
            goal = process_pointcloud_batch(goal)

            token_seq = []
            for i, seq in enumerate(action_str):
                tok = prepare_sequence_tokens(seq.split(' '), language.skill2index)
                token_seq.append(tok)   
            
            # action_tokens = torch.stack(token_seq, 0).to(dev)         
            action_tokens = torch.nn.utils.rnn.pad_sequence(token_seq, batch_first=True).to(dev)

            # convert action sequences to one-hot
            s = action_tokens.size()
            action_tokens = action_tokens.long().to(dev)
            action_tokens_oh = torch.FloatTensor(s[0], s[1], len(language.skill2index)).to(dev)
            action_tokens_oh.zero_()
            action_tokens_oh.scatter_(2, action_tokens[:, :, None], 1)

            reward = torch.zeros((s[0], s[1]+1, 1))
            reward = reward.float().to(dev)

            mask_gt = mask.float().to(dev)
            mask_gt = mask_gt[:, None, :, :].repeat((1, s[1], 1, 1))            

            # initialize GRU hidden state
            batch_size, h_size = observations.size(0), model.hidden_init.size(-1)
            init_h = torch.zeros(batch_size, 1, h_size).to(dev)            

            table_plane = trimesh.creation.box([0.6, 1.0, 0.002])
            table_transformation = np.eye(4)
            table_transformation[:-1, -1] = [0.3, 0.0, 0.0]
            table_plane.apply_transform(table_transformation)
            table_plane.visual.face_colors[:, -1] = 50
            r_loss = 0
            for b_idx in range(batch_size):
                
                o = observations[b_idx][None, :, :].repeat((s[1], 1, 1))
                a, h, m, t, g = action_tokens_oh[b_idx], init_h[b_idx], mask_gt[b_idx], transformation[b_idx], goal[b_idx]
                z_post, z_post_logits, z_prior, z_prior_logits, x_mask, h, reward_pred = model.forward_loop(
                    o.unsqueeze(0), a.unsqueeze(0), h.unsqueeze(0), m.unsqueeze(0), t.unsqueeze(0), g.unsqueeze(0), evaluate=True
                )    

                reward[:, (reward_step[b_idx] - 1):] = 1
                reward_loss = model.mse(reward_pred.squeeze(), reward[b_idx].squeeze())
                r_loss += reward_loss.item()
                # print('Reward Loss: %.3f' % reward_loss, 'Reward Sequence: ', np.around(reward_pred.squeeze().cpu().numpy(), 3).tolist())

                pcd = raw_observations[b_idx].cpu().numpy().squeeze()
                tpcd = trimesh.PointCloud(pcd)
                bbox = tpcd.bounding_box_oriented
                bbox.visual.face_colors[:, -1] = 30                
                for i in range(s[1]):
                    global_step += 1
                    # pcd = pcds[i].cpu().numpy().squeeze()

                    mask = x_mask[0, i].detach().cpu().numpy().squeeze()
                    top_inds = np.argsort(mask, 0)[::-1]
                    pred_mask = np.zeros((mask.shape[0]), dtype=bool)
                    pred_mask[top_inds[:15]] = True

                    imgs = []
                    if tmesh:
                        # ttable = trimesh.PointCloud(tables[b_idx].detach().cpu().data.numpy().squeeze())
                        # ttable.colors = np.tile(np.array([255, 0, 0, 255]), (ttable.vertices.shape[0], 1))   
                        tpcd.colors = np.tile(np.array([255, 0, 255, 10]), (tpcd.vertices.shape[0], 1))
                        tpcd.colors[np.where(pred_mask)[0]] = np.tile(np.array([0, 0, 255, 255]), (np.where(pred_mask)[0].shape[0], 1))  

                        # hand tuned camera angle that looks from in front and above the table
                        scene = trimesh.Scene()
                        scene.add_geometry([tpcd, bbox, table_plane])                        
                        distance = [1.51685814, -0.04729293, 0.54116778]
                        euler = [1.28203444, -0.00225494, 1.53850116]
                        scene.set_camera(angles=euler, distance=distance, center=[0.0, -0.1, 0.3])

                        img = scene.save_image(resolution=(640, 480), visible=True)
                        rendered = Image.open(BytesIO(img)).convert("RGB")
                        np_img = np.array(rendered)
                        torch_img = torch.from_numpy(np_img).permute(2,0,1)
                        writer.add_image('test/mask_image/{}_{}_{}'.format(it, k, b_idx), torch.from_numpy(np_img).permute(2,0,1), i)

            # print('Loss: %.3f' % loss)
            writer.add_scalar('loss/test/reward_loss', r_loss)    


def train(dataloader, test_dataloader, model, optimizer, language, logdir, writer, args):
    model = model.train()

    if args.cuda:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')
    
    it = int(args.resume_iter)
    for epoch in range(1, args.num_epoch):
        for sample in dataloader:
            it += 1
            observations, action_str, tables, mask, reward, transformation, goal = sample

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

            token_seq = []
            for i, seq in enumerate(action_str):
                tok = prepare_sequence_tokens(seq.split(' '), language.skill2index)
                token_seq.append(tok)   
            
            action_tokens = torch.stack(token_seq, 0).to(dev)         
            # action_tokens = torch.nn.utils.rnn.pad_sequence(token_seq, batch_first=True).to(dev)

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

            # predict rewards/probability of reaching the goal -- TODO
            reward_loss = model.mse(reward_pred, reward)

            # predict reward losses -- TODO

            optimizer.zero_grad()
            loss = mask_loss + reward_loss + kl_loss
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss/train/reward_loss', reward_loss.item(), it)
            writer.add_scalar('loss/train/mask_loss', mask_loss.item(), it)
            writer.add_scalar('loss/train/kl_loss', kl_loss.item(), it)

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
                # eval(dataloader, model, language, logdir, writer, args, tmesh=args.tmesh)
                eval_reward(test_dataloader, model, language, logdir, writer, args, it, tmesh=args.tmesh)
                model = model.train()


def main(args):
    
    dataset = SkeletonTransitionDataset(train=True)
    reward_dataset = SkeletonDataset(train=False)
    train_loader = DataLoader(dataset, batch_size=args.batch_size)
    reward_loader = DataLoader(reward_dataset, batch_size=args.batch_size)

    skill_lang = SkillLanguage('default')
    language_loader = DataLoader(dataset, batch_size=1)
    for sample in language_loader:
        seq = sample[1]
        skill_lang.add_skill_seq(seq[0])
    print('Skill Language: ')
    print(skill_lang.skill2index, skill_lang.index2skill)
    
    # skill_lang = dataset.skill_lang
    transition_model = TransitionModelSeqBatch(
        observation_dim=6,
        action_dim=len(skill_lang.skill2index),
        latent_dim=256,
        out_dim=256,
        cat_dim=32)

    if torch.cuda.is_available() and args.cuda:
        dev = torch.device('cuda:0')
        transition_model.cuda()
    else:
        dev = torch.device('cpu')   

    optimizer = torch.optim.Adam(transition_model.parameters(), lr=args.lr)


    logdir = osp.join(args.logdir, args.exp)
    rundir = osp.join(args.rundir, args.exp)
    if not osp.exists(logdir):
        os.makedirs(logdir)
    if not osp.exists(rundir):
        os.makedirs(rundir)
    writer = SummaryWriter(rundir)

    if args.resume_iter != 0:
        model_path = osp.join(logdir, "model_{}".format(args.resume_iter))
        checkpoint = torch.load(model_path)
        args_old = checkpoint['FLAGS']

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        transition_model.load_state_dict(checkpoint['model_state_dict'])

    train(train_loader, reward_loader, transition_model, optimizer, skill_lang, logdir, writer, args)   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='whether or not to train')
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not')

    # Generic Parameters for Experiments
    parser.add_argument('--dataset', default='intphys', type=str, help='intphys or others')
    parser.add_argument('--logdir', default='cachedir', type=str, help='location where log of experiments will be stored')
    parser.add_argument('--rundir', default='runs')
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

    parser.add_argument('--tmesh', action='store_true')

    args = parser.parse_args()

    main(args)