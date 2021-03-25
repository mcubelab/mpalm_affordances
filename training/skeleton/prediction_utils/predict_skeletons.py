import os, os.path as osp
import sys
import argparse
import random
import copy
import time
import signal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

# from airobot.utils import common

sys.path.append(osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives'))
from helper import util2 as util

from data import SkillDataset, SkeletonDataset
from models import MultiStepDecoder, InverseModel
from utils import SkillLanguage, prepare_sequence_tokens


PAD_token = 0
SOS_token = 1
EOS_token = 2


def signal_handler(sig, frame):
    print('Exit')
    sys.exit(0)


def make_prediction(args, obs_file, model, inverse_model, language_index2skill, language_skill2index):
    model.eval()
    inverse_model.eval()

    if args.cuda:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    model = model.to(dev)
    inverse_model = inverse_model.to(dev)

    while True:
        try:
            observation = np.load(osp.join(args.observation_dir, obs_file), allow_pickle=True)
            break
        except:
            pass
        time.sleep(0.01)
    point_cloud = observation['pointcloud_pts'][:100]
    next_point_cloud = observation['next_pointcloud_pts'][:100]

    pcd_mean = np.mean(point_cloud, axis=0, keepdims=True)
    point_cloud = np.concatenate((point_cloud - pcd_mean, np.tile(pcd_mean, (point_cloud.shape[0], 1))), axis=1)
    next_pcd_mean = np.mean(next_point_cloud, axis=0, keepdims=True)
    next_point_cloud = np.concatenate((next_point_cloud - next_pcd_mean, np.tile(next_pcd_mean, (next_point_cloud.shape[0], 1))), axis=1)    
    transformation = observation['transformation']

    # make prediction
    with torch.no_grad():
        point_cloud = torch.from_numpy(point_cloud).float().to(dev)[None, :, :]
        next_point_cloud = torch.from_numpy(next_point_cloud).float().to(dev)[None, :, :]
        subgoal = torch.from_numpy(transformation).float().to(dev)[None, :]
        task_emb = inverse_model(point_cloud, next_point_cloud, subgoal)

        decoder_input = torch.Tensor([[SOS_token]]).long().to(dev)
        decoder_hidden = task_emb[None, :]

        decoded_skills = []
        for t in range(args.max_seq_length):
            decoder_input = model.embed(decoder_input)
            decoder_output, decoder_hidden = model.gru(decoder_input, decoder_hidden)
            output = model.log_softmax(model.out(decoder_output[:, 0]))
            topv, topi = output.topk(1, dim=1)
            decoder_input = topi

            if topi.item() == language_skill2index['EOS']:
                decoded_skills.append('EOS')
                break
            else:
                decoded_skills.append(language_index2skill[topi.item()])
        print('decoded: ', decoded_skills)
        print('\n')

    # TODO process decoded skills

    # write outputs to filesystem
    pred_fname = osp.join(args.prediction_dir, obs_file)
    print('making prediction to: ' + str(pred_fname))
    np.savez(
        pred_fname,
        skeleton=decoded_skills
    )
    os.remove(osp.join(args.observation_dir, obs_file))


def main(args):
    model_path = osp.join(
        args.logdir,
        args.model_path,
        'model_' + str(args.model_number))

    print('Loading from model path: ' + str(model_path))
    checkpoint = torch.load(model_path)
    args_old = checkpoint['args']
    language_skill2index = checkpoint['language_skill2idx']
    language_index2skill = checkpoint['language_idx2skill']

    # setup skeleton prediction model
    in_dim = 6
    # out_dim = 9
    hidden_dim = args.latent_dim
    out_dim = len(language_index2skill.keys())

    inverse = InverseModel(in_dim, hidden_dim, hidden_dim)
    model = MultiStepDecoder(hidden_dim, out_dim)



    model.load_state_dict(checkpoint['model_state_dict'])
    inverse.load_state_dict(checkpoint['inverse_model_state_dict'])
    # try:
    # except:
    #     print('NOT LOADING PRETRAINED WEIGHTS!!!')

    # set up obs/pred dirs
    if not osp.exists(args.prediction_dir):
        os.makedirs(args.prediction_dir)
    if not osp.exists(args.observation_dir):
        os.makedirs(args.observation_dir)

    model = model.eval()
    inverse = inverse.eval()
    signal.signal(signal.SIGINT, signal_handler)

    print('Starting loop')
    while True:
        obs_fnames = os.listdir(args.observation_dir)
        observation_available = len(obs_fnames) > 0
        if observation_available:
            print('Observation available')
            for fname in obs_fnames:
                if fname.endswith('.npz'):
                    time.sleep(0.5)
                    make_prediction(args, fname, model, inverse, language_index2skill, language_skill2index)
        time.sleep(0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--observation_dir', type=str, default='/tmp/skeleton/observations')
    parser.add_argument('--prediction_dir', type=str, default='/tmp/skeleton/predictions')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_number', type=int, default=20000)

    parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not')

    # Generic Parameters for Experiments
    parser.add_argument('--logdir', default='vae_cachedir', type=str, help='location where log of experiments will be stored')
    parser.add_argument('--exp', default='debug', type=str, help='name of experiments')

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
