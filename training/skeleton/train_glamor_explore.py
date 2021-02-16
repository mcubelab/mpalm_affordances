import os, os.path as osp
import sys
import argparse
import random
import copy
import signal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

# from airobot.utils import common

from data import SkillPlayDataset, SkeletonDatasetGlamor
from glamor.models import MultiStepDecoder, InverseModel
from lcm_inference.skeleton_predictor_lcm import GlamorSkeletonPredictorLCM
from skeleton_utils.utils import SkillLanguage, prepare_sequence_tokens, process_pointcloud_batch
from skeleton_utils.skeleton_globals import PAD_token, SOS_token, EOS_token
from skeleton_utils.replay_buffer import TransitionBuffer
from skeleton_utils.buffer_lcm import BufferLCM
from train_glamor import train as pretrain


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


def train(model, inverse_model, buffer, optimizer, language, args, logdir):
    model.train()
    inverse_model.train()

    if args.cuda:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    model = model.to(dev)
    inverse_model = inverse_model.to(dev)

    iterations = 0
    for epoch in range(args.num_epoch):
        sample = buffer.sample_sg(n=args.batch_size)
        # for sample in buffer_samples:
        iterations += 1

        from IPython import embed
        embed()
        subgoal, contact, observation, next_observation, action_seq = sample

        bs = subgoal.size(0)
        
        observation = observation.float().to(dev)
        next_observation = next_observation.float().to(dev)
        observation = process_pointcloud_batch(observation).to(dev)
        next_observation = process_pointcloud_batch(next_observation).to(dev)
        subgoal = subgoal.float().to(dev)
        task_emb = inverse_model(observation, next_observation, subgoal)
        # padded_seq_batch = torch.nn.utils.rnn.pad_sequence(token_seq, batch_first=True).to(dev)
        padded_seq_batch = action_seq
        print('REMOVE THIS!!! clipping action tokens for testing')
        padded_seq_batch = torch.clamp(padded_seq_batch, 0, len(language.skill2index.keys())).squeeze().to(dev)
        
        decoder_input = torch.Tensor([[SOS_token]]).repeat((padded_seq_batch.size(0), 1)).long().to(dev)
        decoder_hidden = task_emb[None, :, :]
        loss = 0
        max_seq_length = padded_seq_batch.size(1)
        for t in range(max_seq_length):
            decoder_input = model.embed(decoder_input)
            decoder_output, decoder_hidden = model.gru(decoder_input, decoder_hidden)
            output = model.log_softmax(model.out(decoder_output[:, 0])).view(bs, -1)
            # loss += model.criterion(output, padded_seq_batch[:, t].squeeze())
            loss += model.criterion(output, padded_seq_batch[:, t])
            topv, topi = output.topk(1, dim=1)
            decoder_input = topi
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iterations % args.log_interval == 0:
            kvs = {}
            kvs['loss'] = loss.item()
            log_str = 'Epoch: {}, Iteration: {}, '.format(epoch, iterations)
            for k, v in kvs.items():
                log_str += '%s: %.5f, ' % (k,v)
            print(log_str)

        if iterations % args.save_interval == 0:
            model = model.eval()
            model_path = osp.join(logdir, "model_{}".format(iterations))
            torch.save({'model_state_dict': model.state_dict(),
                        'inverse_model_state_dict': inverse_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'language_idx2skill': language.index2skill,
                        'language_skill2idx': language.skill2index,
                        'args': args}, model_path)
            print("Saving model in directory....")


def main(args):
    signal.signal(signal.SIGINT, signal_handler)
    buffer = TransitionBuffer(
        size=5000,
        observation_n=(100, 3),
        action_n=1,
        device=torch.device('cuda:0'),
        goal_n=7)
        

    buffer_to_lcm = BufferLCM(buffer)

    train_data = SkeletonDatasetGlamor('train')
    test_data = SkeletonDatasetGlamor('test') 
    # train_data = SkeletonDataset('overfit')
    # test_data = SkeletonDataset('overfit')      

    skill_lang = SkillLanguage('default')

    language_loader = DataLoader(train_data, batch_size=1)
    for sample in language_loader:
        # seq = sample[1]
        seq = sample[-1]
        skill_lang.add_skill_seq(seq[0])
    print('Skill Language: ')
    print(skill_lang.skill2index, skill_lang.index2skill)


    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    
    hidden_dim = args.latent_dim

    # in_dim is [x, y, z, x0, y0, z0]
    # out_dim mukst include total number of skills, pad token, and eos token
    in_dim = 6
    # out_dim = 9
    # out_dim = len(skill_lang.skill2index.keys())
    out_dim = len(skill_lang.index2skill.keys())

    inverse = InverseModel(in_dim, hidden_dim, hidden_dim).cuda()
    model = MultiStepDecoder(hidden_dim, out_dim).cuda()
    params = list(inverse.parameters()) + list(model.parameters())

    optimizer = optim.Adam(params, lr=args.lr)

    logdir = osp.join(args.logdir, args.exp)
    if not osp.exists(logdir):
        os.makedirs(logdir)

    if args.resume_iter != 0:
        model_path = osp.join(logdir, "model_{}".format(args.resume_iter))
        checkpoint = torch.load(model_path)
        args_old = checkpoint['args']
        for kwarg, value in args.__dict__.items():
            if kwarg not in args_old.__dict__.keys():
                args_old.__dict__[kwarg] = value 
            if 'pretrain' not in kwarg:
                args.__dict__[kwarg] = args_old.__dict__[kwarg]
        # args = checkpoint['args']
        # args = args_old

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        inverse.load_state_dict(checkpoint['inverse_model_state_dict'])

        # skill_lang.index2skill = checkpoint['language_idx2skill']
        # skill_lang.skill2index = checkpoint['language_skill2idx']
            
    if args.pretrain:
        print('pretraining')
        args.num_epoch = copy.deepcopy(args.num_pretrain_epoch)
        pretrain(model, inverse, train_loader, test_loader, optimizer, skill_lang, args, logdir)

    lcm_predictor = GlamorSkeletonPredictorLCM(
        model=model,
        inverse_model=inverse,
        args=args,
        language=skill_lang)

    print('beginning LCM loop')
    while True:
        try:
            # send some data over 
            new_vals = 0
            model = model.eval()
            while True:
                print('predicting')
                predict_val = lcm_predictor.predict_skeleton()
            
                # add the data to the buffer
                while True:
                    print('adding to buffer')
                    received_val = buffer_to_lcm.receive_and_append_buffer()
                    if received_val:
                        break

                new_vals += 1
                if new_vals > args.env_episodes_to_add:
                    break
            
            # train the models
            print('training')
            model = model.train()
            args.num_epoch = copy.deepcopy(args.num_train_epoch)
            train(model, inverse, buffer, optimizer, skill_lang, args, logdir)
        except KeyboardInterrupt:
            pass

    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true', help='whether or not to pretrain on supervised data')
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not') 
 
    # Generic Parameters for Experiments
    parser.add_argument('--logdir', default='glamor_rl_cachedir', type=str, help='location where log of experiments will be stored')
    parser.add_argument('--rundir', default='runs/glamor_rl_runs')
    parser.add_argument('--exp', default='glamor_debug', type=str, help='name of experiments')
    parser.add_argument('--data_workers', default=4, type=int, help='Number of different data workers to load data in parallel')

    # train
    parser.add_argument('--batch_size', default=16, type=int, help='size of batch of input to use')
    parser.add_argument('--num_pretrain_epoch', default=10, type=int, help='number of epochs of training to run')
    parser.add_argument('--num_train_epoch', default=100, type=int, help='number of epochs of training to run')
    parser.add_argument('--num_epoch', default=100, type=int, help='number of epochs of training to run')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for training')
    parser.add_argument('--log_interval', default=10, type=int, help='log outputs every so many batches')
    parser.add_argument('--save_interval', default=100, type=int, help='save outputs every so many batches')
    parser.add_argument('--resume_iter', default=0, type=str, help='iteration to resume training')
    parser.add_argument('--env_episodes_to_add', default=10, type=int, help='number of samples to add to the replay buffer before we train again')

    # model 
    parser.add_argument('--latent_dim', default=512, type=int, help='size of hidden representation')
    parser.add_argument('--max_seq_length', default=5, type=int, help='maximum sequence length')

    args = parser.parse_args()

    main(args)