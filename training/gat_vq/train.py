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

from data import RobotDatasetGrasp, RobotDataset
from models import EnergyModel
from random import choices
# from apex.optimizers import FusedAdam
from torch.optim import Adam
import argparse
from itertools import permutations
import itertools
import matplotlib.pyplot as plt
# from baselines.logger import TensorBoardOutputFormat



"""Parse input arguments"""
parser = argparse.ArgumentParser(description='Train reasoning model')
parser.add_argument('--train', action='store_true', help='whether or not to train')
parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not')

# Generic Parameters for Experiments
parser.add_argument('--dataset', default='intphys', type=str, help='intphys or others')
parser.add_argument('--logdir', default='cachedir', type=str, help='location where log of experiments will be stored')
parser.add_argument('--exp', default='debug', type=str, help='name of experiments')
parser.add_argument('--data_workers', default=16, type=int, help='Number of different data workers to load data in parallel')

# train
parser.add_argument('--batch_size', default=256, type=int, help='size of batch of input to use')
parser.add_argument('--num_epoch', default=10000, type=int, help='number of epochs of training to run')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for training')
parser.add_argument('--log_interval', default=10, type=int, help='log outputs every so many batches')
parser.add_argument('--save_interval', default=10000, type=int, help='save outputs every so many batches')
parser.add_argument('--resume_iter', default=0, type=str, help='iteration to resume training')

parser.add_argument('--vis', action='store_true', help='vis')

# Distributed training hyperparameters
parser.add_argument('--nodes', default=1, type=int, help='number of nodes for training')
parser.add_argument('--gpus', default=1, type=int, help='number of gpus per nodes')
parser.add_argument('--node_rank', default=0, type=int, help='rank of node')





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
    counter = 0

    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    with torch.no_grad():
        for start, goal, contact_obj_frame, contact_obj_frame_neg, obj_id in test_dataloader:
            batch_size = start.size(0)

            start = start.to(dev).float()
            goal = goal.to(dev).float()
            contact_obj_frame = contact_obj_frame.to(dev).float()

            contact_dim = contact_obj_frame.size(1)

            inp = torch.cat([start, goal, contact_obj_frame], dim=1)

            idx = torch.randint(0, batch_size, (inp.size(0),))
            idx = idx[:, None].repeat(1, inp.size(1)).to(dev)

            inp_neg = torch.gather(inp, 0, idx)

            idx = torch.randint(0, batch_size, (inp.size(0),))
            idx = idx[:, None].repeat(1, contact_dim).to(dev)

            contact_context = torch.gather(contact_obj_frame, 0, idx)
            contact_context_repeat = contact_context[None, :, :].repeat(inp.size(0), 1, 1)

            joint_inp_neg = torch.cat([inp[:, None, :-contact_dim].repeat(1, contact_context.size(0), 1), contact_context_repeat], dim=2)
            s = joint_inp_neg.size()
            inp_neg = joint_inp_neg.view(-1, s[2])
            energy_neg = model.forward(inp_neg)

            # for i in range(10):
            #     contact_context_repeat = contact_context_repeat.detach()
            #     contact_context_repeat.requires_grad = True

            #     joint_inp_neg = torch.cat([inp[:, None, :-contact_dim].repeat(1, contact_context.size(0), 1), contact_context_repeat], dim=2)
            #     s = joint_inp_neg.size()
            #     inp_neg = joint_inp_neg.view(-1, s[2])
            #     energy_neg = model.forward(inp_neg)
            #     contact_context_repeat_grad = torch.autograd.grad([energy_neg.sum()], [contact_context_repeat])[0]
            #     contact_context_repeat = contact_context_repeat - 0.01 * contact_context_repeat_grad
                # contact_context_repeat = clip_to_reasonable(contact_context_repeat)

            joint_inp_neg = joint_inp_neg.cpu().detach().numpy()
            inp_neg = inp_neg.cpu().detach().numpy()
            energy_neg = energy_neg.cpu().detach().numpy()

            for i in range(joint_inp_neg.shape[0]):
                energy_neg = energy_neg.reshape((s[0], s[1]))
                inp_neg_i = joint_inp_neg[i]

                energy_idx = np.argsort(energy_neg[i])
                for idx in range(5):
                    output_file = osp.join("output", "state_{}_{}.npz".format(counter, idx))
                    data = inp_neg_i[energy_idx[idx]]

                    output_dict = {}
                    np.savez(output_file, data=data)

                counter += 1

            if counter > 100:
                break

def clip_to_reasonable(inp):
    if len(inp.size()) == 2:
        inp[:, 3:7] = F.normalize(inp[:, 3:7], p=2, dim=1)
        inp[:, 10:14] = F.normalize(inp[:, 10:14], p=2, dim=1)
    elif len(inp.size()) == 3:
        inp[:, :, 3:7] = F.normalize(inp[:, :, 3:7], p=2, dim=1)
        inp[:, :, 10:14] = F.normalize(inp[:, :, 10:14], p=2, dim=1)

    return inp

def train(train_dataloader, test_dataloader, model, optimizer, FLAGS, logdir, rank_idx):
    it = int(FLAGS.resume_iter)
    optimizer.zero_grad()

    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    model = model.to(dev).train()


    n_sample = 1000
    for epoch in range(FLAGS.num_epoch):
        for start, goal, contact_obj_frame, contact_obj_frame_neg in train_dataloader:
            batch_size = start.size(0)

            start = start.to(dev).float()
            goal = goal.to(dev).float()
            contact_obj_frame = contact_obj_frame.to(dev).float()
            contact_obj_frame_neg = contact_obj_frame_neg.to(dev).float()

            contact_dim = contact_obj_frame.size(1)

            inp = torch.cat([start, goal, contact_obj_frame], dim=1)

            idx = torch.randint(0, batch_size, (inp.size(0),))
            idx = idx[:, None].repeat(1, inp.size(1)).to(dev)
            inp_neg = torch.gather(inp, 0, idx)
            idx = torch.randint(0, batch_size, (inp.size(0),))
            idx = idx[:, None].repeat(1, contact_dim).to(dev)
            contact_obj_frame_neg = torch.gather(contact_obj_frame, 0, idx)

            # for i in range(10):
            #     contact_obj_frame_neg = contact_obj_frame_neg.detach()
            #     contact_obj_frame_neg.requires_grad = True
            #     inp_neg = torch.cat([inp[:, :-contact_dim], contact_obj_frame_neg], dim=1)
            #     energy_neg = model.forward(inp_neg)
            #     contact_obj_grad = torch.autograd.grad([energy_neg.sum()], [contact_obj_frame_neg])[0]
            #     contact_obj_frame_neg = contact_obj_frame_neg - 0.01 * contact_obj_grad
            #     contact_obj_frame_neg = clip_to_reasonable(contact_obj_frame_neg)


            inp_neg = torch.cat([inp[:, :-contact_dim], contact_obj_frame_neg.detach()], dim=1)
            energy_neg = model.forward(inp_neg)
            energy_pos = model.forward(inp)

            partition = torch.cat([energy_pos[:, None, :], energy_neg[None, :, :].repeat(batch_size, 1, 1)], dim=1)


            log_prob = (-energy_pos) - torch.logsumexp(-partition, dim=1)
            loss = (-log_prob).mean()
            it += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            if it % FLAGS.log_interval == 0 and rank_idx == 0:
                loss = loss.item()

                kvs = {}
                kvs['energy_pos'] = energy_pos.mean().item()
                kvs['energy_neg'] = energy_neg.mean().item()
                # kvs['grad_mag'] = torch.abs(contact_obj_grad).mean().item()
                kvs['loss'] = loss
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
    dataset_train = RobotDatasetGrasp('train')
    dataset_test = RobotDatasetGrasp('test')

    train_dataloader = DataLoader(dataset_train, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=False, drop_last=False)
    test_dataloader = DataLoader(dataset_test, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False, drop_last=False)

    ## model
    model = EnergyModel().to(device)
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
