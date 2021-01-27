import os, os.path as osp
import sys
import argparse
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.append(osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives'))
sys.path.append(osp.join(os.environ['CODE_BASE'], 'training/gat'))
from helper import util2 as util
# from models_vae import GeomEncoder
from gat_dgl import GeomEncoder
from dreamer_utils import to_onehot, to_onehot3d


class TransitionModelSeqBatch(nn.Module):
    def __init__(self, observation_dim, action_dim, latent_dim, out_dim, cat_dim=32):
        super(TransitionModelSeqBatch, self).__init__()
        # dimensionality of the categorical variable (assume that we will use cat_dim variables each of cat_dim dimension)
        self.cat_dim = cat_dim

        self.encoder_fc1 = nn.Linear(2*latent_dim, 512)
        self.encoder_fc2 = nn.Linear(512, cat_dim**2)

        self.transition_fc1 = nn.Linear(latent_dim, 512)
        self.transition_fc2 = nn.Linear(512, cat_dim**2)


        # self.pcd_decoder = GeomEncoder(observation_dim + cat_dim**2 + latent_dim, latent_dim)
        self.pcd_decoder = GeomEncoder(observation_dim, latent_dim)
        self.decoder_fc1 = nn.Linear(latent_dim*2 + cat_dim**2, latent_dim)
        self.mask = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1), nn.Sigmoid())

        self.gru = nn.GRU(cat_dim**2 + action_dim, latent_dim, batch_first=True)

        self.encoder = GeomEncoder(observation_dim, latent_dim)
        
        # initialize the hidden state.
        self.hidden_init = torch.randn(1, 1, latent_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.bce = nn.BCELoss()

    def forward(self, x_t, a_t, h_t):
        # encode point cloud observation at time t-1 from h at time t-1
        z_posterior_logits_t, z_posterior_t = self.encode(x_t, h_t)

        # reconstruct part of the input at time t-1
        x_mask_t = self.decode(x_t, h_t, z_posterior_t.view(-1, self.cat_dim**2))

        # run forward dynamics, to get stochastic and deterministic latent states at time t, from h, z, and a at time t-1
        out_t1, h_t1, z_prior_logits_t1, z_prior_t1 = self.forward_dynamics(h_t, z_posterior_t.view(-1, self.cat_dim**2), a_t)
        return z_posterior_logits_t, z_posterior_t, x_mask_t, h_t1, z_prior_logits_t1, z_prior_t1

    def forward_loop(self, x, a, h):
        T = x.size(1) + 1
        z_post_list = [torch.empty(0)]*T
        z_prior_list = [torch.empty(0)]*T
        z_post_logits_list = [torch.empty(0)]*T
        z_prior_logits_list = [torch.empty(0)]*T
        h_list = [torch.empty(0)]*T
        x_mask_list = [torch.empty(0)]*T

        h_list[0] = h
        for t in range(T-1):
            z_logits, z = self.encode(x[:, t], h_list[t])

            x_mask = self.decode(x[:, t], h_list[t], z.view(-1, self.cat_dim**2))

            _, h_next, z_hat_logits, z_hat = self.forward_dynamics(h_list[t], z.view(-1, self.cat_dim**2), a[:, t])

            z_post_list[t] = z
            z_post_logits_list[t] = z_logits
            x_mask_list[t] = x_mask
            z_prior_list[t+1] = z_hat
            z_prior_logits_list[t+1] = z_hat_logits
            h_list[t+1] = h_next

        # perform the autoencoding of the last step, where the object has reached the goal
        z_logits, z = self.encode(x[:, -1], h_next)
        x_mask = self.decode(x[:, -1], h_next, z.view(-1, self.cat_dim**2))            
        z_post_list[-1] = z
        z_post_logits_list[-1] = z_logits
        x_mask_list[-1] = x_mask

        # TODO -- figure out what to put in for the first entry of the prior? no dynamics before first step...
        z_prior_list[0] = z_post_list[0].clone()
        z_prior_logits_list[0] = z_post_logits_list[0].clone()

        z_post, z_post_logits = torch.stack(z_post_list, 1), torch.stack(z_post_logits_list, 1)
        z_prior, z_prior_logits = torch.stack(z_prior_list, 1), torch.stack(z_prior_logits_list, 1)
        x_mask, h = torch.stack(x_mask_list, 1), torch.stack(h_list, 1)

        return z_post, z_post_logits, z_prior, z_prior_logits, x_mask, h

    def encode(self, x, h):
        # get point cloud embedding, per points
        x = self.encoder(x)

        # average pool over all point embeddings
        x = x.mean(dim=1)

        # combine pcd emb and hidden state
        x = torch.cat((x, h.view(-1, x.size(1))), dim=1)

        # obtain categorical logits and get sample with straight-thru gradient
        x = self.encoder_fc2(F.relu(self.encoder_fc1(x)))
        x = x.view(-1, self.cat_dim, self.cat_dim)
        m = Categorical(logits=x)
        sample = to_onehot3d(m.sample()[:, :, None], self.cat_dim)
        sample = sample + self.softmax(x) - self.softmax(x).detach()
        return x, sample

    def decode(self, x, h, z):
        # # combine latent and original point cloud features    
        z = z[:, None, :].repeat((1, x.size(1), 1))
        h = h.repeat((1, x.size(1), 1))
        # decoder_input = torch.cat((x, z, h), dim=2)

        # # obtain per-point features from GAT
        # latent = self.pcd_decoder(decoder_input)
        point_latent = self.pcd_decoder(x)
        latent = torch.cat((point_latent, z, h), dim=2)
        latent = self.decoder_fc1(latent)

        # classify contact or no-contact
        mask = self.mask(latent)
        return mask

    def forward_recurrent(self, h, z, a):   
        # combine z state and one-hot action embedding       
        x = torch.cat((z, a), dim=1)[:, None, :]

        # one-step of recurrent model
        output, hidden = self.gru(x, h.permute(1, 0, 2))  # why do we need this permute?
        return output, hidden.permute(1, 0, 2)

    def forward_transition(self, h):
        # get logits for categorical purely from h and sample
        x = self.transition_fc2(F.relu(self.transition_fc1(h)))
        x = x.view(-1, self.cat_dim, self.cat_dim)
        m = Categorical(logits=x)
        sample = to_onehot3d(m.sample()[:, :, None], self.cat_dim)
        sample = sample + self.softmax(x) - self.softmax(x).detach()
        return x, sample

    def forward_dynamics(self, h, z, a):
        # get deterministic hidden state from forward dynamics
        output, hidden = self.forward_recurrent(h, z, a)

        # use this to also get a stochastic sample
        logits, z_sample = self.forward_transition(hidden)
        return output, hidden, logits, z_sample

