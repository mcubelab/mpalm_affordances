import argparse
import torch
import copy
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch_geometric.nn import GATConv
import numpy as np
from quantizer import VectorQuantizer


class GeomEncoder(nn.Module):

    def __init__(self,
                 in_dim,
                 latent_dim,
                 table_mesh=False):
        super(GeomEncoder, self).__init__()
        inner_dim = 256

        if table_mesh:
            max_size = 125
        else:
            max_size = 100

        self.inner_dim = inner_dim
        self.max_size = max_size

        self.remap = nn.Linear(in_dim, inner_dim)
        self.conv1 = GATConv(inner_dim, inner_dim)
        self.conv2 = GATConv(inner_dim, inner_dim)
        self.conv3 = GATConv(inner_dim, inner_dim)
        self.conv4 = GATConv(inner_dim, latent_dim)

        edge_idx = 1 - np.tri(max_size)
        grid_x, grid_y = np.arange(max_size), np.arange(max_size)
        grid_x, grid_y = np.tile(grid_x[:, None], (1, max_size)), np.tile(grid_y[None, :], (max_size, 1))
        self.pos = np.stack([grid_x, grid_y], axis=2).reshape((-1, 2))
        self.default_mask = edge_idx.astype(np.bool)

    def forward(self, x, full=False):
        pos = x[:, :, :3]
        x = F.relu(self.remap(x))
        assert (x.size(1) == self.max_size)

        if full:
            default_mask = self.default_mask
            mask = np.ones((x.size(0), self.max_size, self.max_size), dtype=np.bool)
            mask_new = mask * default_mask[None, :, :]
            mask_new = mask_new.reshape((-1, self.max_size ** 2))

            edge_idxs = [self.pos + self.max_size * i for i in range(x.size(0))]
            edge_idxs = np.concatenate(edge_idxs, axis=0)
            edge = torch.LongTensor(edge_idxs).transpose(0, 1)
            edge = edge.to(x.device)
        else:
            diff = torch.norm(pos[:, :, None, :] - pos[:, None, :, :], p=2, dim=3)
            tresh = diff < 0.03
            mask = tresh.detach().cpu().numpy()

            default_mask = self.default_mask
            mask_new = mask.astype(np.bool) * default_mask[None, :, :]
            mask_new = mask_new.reshape((-1, self.max_size ** 2))

            edge_idxs = [self.pos[mask_new[i]] + self.max_size * i for i in range(x.size(0))]
            edge_idxs = np.concatenate(edge_idxs, axis=0)
            edge = torch.LongTensor(edge_idxs).transpose(0, 1)
            edge = edge.to(x.device)

        s = x.size()
        x = x.view(-1, s[2])

        x = F.relu(self.conv1(x, edge) + x)
        x = F.relu(self.conv2(x, edge) + x)
        x = F.relu(self.conv3(x, edge) + x)
        x = self.conv4(x, edge)

        x = x.view(s[0], s[1], -1)

        return x
