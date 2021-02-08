import os, os.path as osp
import time
import numpy as np
import sys
import signal
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
import torch.nn as nn

from joint_model_vae import JointPointVAE, JointVAEFull
from lcm_inference.skill_sampler_lcm import ModelPredictorLCM

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
parser.add_argument('--pointnet', action='store_true')
parser.add_argument('--primitive_name', type=str, default='grasp')
parser.add_argument('--local_graph', action='store_true')

parser.add_argument('--model_path', type=str)
parser.add_argument('--model_number', type=int, default=20000)
# parser.add_argument('--dgl', action='store_true')
parser.add_argument('--gnn_library', type=str, default='dgl')


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


def main_single(rank, FLAGS):
    if FLAGS.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_dim = 14
    output_dim = 7
    decoder_inp_dim = 7

    # check which GNN library to use
    gnn_libs = {
        'pytorch-geometric': ['pyg', 'pytorch-geometric'],
        'deep-graph-library': ['dgl', 'deep-graph-library']
    }
    gnn_lib_options = [y for x in gnn_libs.values() for y in x]
    gnn_lib = FLAGS.gnn_library
    if gnn_lib not in gnn_lib_options:
        raise ValueError('GNN library not recognized, exiting')    
    
    if gnn_lib in gnn_libs['pytorch-geometric']:
        use_pyg = True
    elif gnn_lib in gnn_libs['deep-graph-library']:
        use_pyg = False

    if FLAGS.pointnet:
        model = JointPointVAE(
            input_dim,
            output_dim,
            FLAGS.latent_dimension,
            decoder_inp_dim,
            hidden_layers=[512, 512],
            pyg=use_pyg
        ).cuda()
    else:
        model = JointVAEFull(
            input_dim,
            output_dim,
            FLAGS.latent_dimension,
            decoder_inp_dim,
            hidden_layers=[512, 512],
            pyg=use_pyg
        ).cuda()        
    
    model_path = osp.join( 
        FLAGS.logdir, 
        FLAGS.model_path, 
        'model_' + str(FLAGS.model_number))

    print('Loading from model path: ' + str(model_path))
    checkpoint = torch.load(model_path)
    FLAGS_OLD = checkpoint['FLAGS']

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print('NOT LOADING PRETRAINED WEIGHTS!!!')

    signal.signal(signal.SIGINT, signal_handler)
    model = model.eval()

    in_msg_name = FLAGS.primitive_name + '_vae_env_observations'
    out_msg_name = FLAGS.primitive_name + '_vae_model_predictions'
    lcm_predictor = ModelPredictorLCM(
        model, FLAGS, model_path, [in_msg_name], out_msg_name 
    )

    try:
        while True:
            lcm_predictor.predict_params()
    except KeyboardInterrupt:
        pass


def main():
    FLAGS = parser.parse_args()
    print(FLAGS)

    main_single(0, FLAGS)


if __name__ == "__main__":
    main()
