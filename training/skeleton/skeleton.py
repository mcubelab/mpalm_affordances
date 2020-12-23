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

from airobot.utils import common

sys.path.append(osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives'))
from helper import util2 as util

from data import SkillDataset
from models import MultiStepDecoder, InverseModel
from utils import SkillLanguage, prepare_sequence_tokens


SOS_token = 0
EOS_token = 1


def eval(model, inverse_model, dataloader, language, args):
    model.eval()
    inverse_model.eval()

    if args.cuda:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    model = model.to(dev)
    inverse_model = inverse_model.to(dev)

    with torch.no_grad()
        for sample in dataloader:
            subgoal, contact, observation, next_observation, action_seq = sample
            action_token_seq = torch.zeros((args.batch_size, args.max_seq_length)).long()
            for i, seq in enumerate(action_seq):
                tok = prepare_sequence_tokens(seq.split(' '), language.skill2index)
                action_token_seq[i, :tok.size(0)] = tok

            action_token_seq = action_token_seq.to(dev)
            observation = observation.float().to(dev)
            next_observation = next_observation.float().to(dev)
            subgoal = subgoal.float().to(dev)

            task_emb = inverse_model(observation, next_observation, subgoal)
            # TODO: figure out how to go over batches    
            for i in range(args.batch_size):
                # prepare input to get point cloud embeddings
                task_embedding = task_emb[i]
                target = action_token_seq[i]

                # begin with SOS
                decoder_input = torch.Tensor([[SOS_token]]).long().to(dev)
                decoder_hidden = task_embedding[None, None, :]
                
                decoded_skills = []
                for _ in range(target.size(0)):
                    decoder_output, decoder_hidden = model(decoder_input, decoder_hidden)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()
                    if topi.item() == language.skill2index['EOS']:
                        decoded_skills.append('EOS')
                        break
                    else:
                        decoded_skills.append(language.index2skill[topi.item()])
                
                print(decoded_skills)


def train(model, inverse_model, dataloader, language, args):
    model.train()
    inverse_model.train()

    if args.cuda:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    model = model.to(dev)
    inverse_model = inverse_model.to(dev)

    for sample in dataloader:
        subgoal, contact, observation, next_observation, action_seq = sample
        action_token_seq = torch.zeros((args.batch_size, args.max_seq_length)).long()
        for i, seq in enumerate(action_seq):
            tok = prepare_sequence_tokens(seq.split(' '), language.skill2index)
            action_token_seq[i, :tok.size(0)] = tok

        action_token_seq = action_token_seq.to(dev)
        observation = observation.float().to(dev)
        next_observation = next_observation.float().to(dev)
        subgoal = subgoal.float().to(dev)

        task_emb = inverse_model(observation, next_observation, subgoal)
        # TODO: figure out how to go over batches    
        for i in range(args.batch_size):
            # prepare input to get point cloud embeddings
            task_embedding = task_emb[i]
            target = action_token_seq[i]

            # begin with SOS
            decoder_input = torch.Tensor([[SOS_token]]).long().to(dev)
            decoder_hidden = task_embedding[None, None, :]
            
            loss = 0

            for t in range(target.size(0)):
                decoder_output, decoder_hidden = model(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += model.criterion(decoder_output, target[t])
            
            print(decoded_skills)

        from IPython import embed
        embed()


def main(args):
    train_data = SkillDataset('train')
    test_data = SkillDataset('test')
    skill_lang = SkillLanguage('default')
    language_loader = DataLoader(train_data, batch_size=1)
    for sample in language_loader:
        seq = sample[-1]
        skill_lang.add_skill_seq(seq[0])
    print('Skill Language: ')
    print(skill_lang.skill2index, skill_lang.index2skill)

    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    
    in_dim = 6
    hidden_dim = args.latent_dim
    out_dim = 6

    inverse = InverseModel(in_dim, hidden_dim, hidden_dim)
    model = MultiStepDecoder(hidden_dim, out_dim)

    if args.train:
        train(model, inverse, train_loader, skill_lang, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='whether or not to train')
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

    # model 
    parser.add_argument('--latent_dim', default=512, type=int, help='size of hidden representation')
    parser.add_argument('--max_seq_length', default=5, type=int, help='maximum sequence length')

    args = parser.parse_args()

    main(args)
