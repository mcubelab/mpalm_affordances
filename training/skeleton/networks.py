import dgl
from dgl import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import time
import numpy as np
from dgl.nn.pytorch import KNNGraph


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
        # x = self.fc3(x)
        # return x


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g

        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attention = nn.Linear(2*out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat((edges.src['z'], edges.dst['z']), dim=1)
        a = self.attention(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # print(edges.src['z'], edges.data['e'])
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)

        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)

        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))

        self.merge = merge

    def forward(self, h):
        head_outs = [attention_head(h) for attention_head in self.heads]

        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GeomEncoder(nn.Module):
    # def __init__(self, g, in_dim, latent_dim, num_heads=1):
    def __init__(self, in_dim, latent_dim, num_heads=1):
        super(GeomEncoder, self).__init__()
        inner_dim = 256
        max_size = 100

        self.g = None

        self.inner_dim = inner_dim
        self.max_size = max_size

        self.remap = nn.Linear(in_dim, inner_dim)
        self.conv1 = MultiHeadGATLayer(self.g, inner_dim, inner_dim, num_heads=num_heads)
        self.conv2 = MultiHeadGATLayer(self.g, inner_dim, inner_dim, num_heads=num_heads)
        self.conv3 = MultiHeadGATLayer(self.g, inner_dim, inner_dim, num_heads=num_heads)
        self.conv4 = MultiHeadGATLayer(self.g, inner_dim, latent_dim, num_heads=num_heads)
        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        # self.layers = [self.conv1, self.conv2]

    def _update_graph(self):
        for layer in self.layers:
            layer.g = self.g
            for head in layer.heads:
                head.g = self.g

    def forward(self, x, *args, **kwargs):
        knn = KNNGraph(x.size(1))
        self.g = knn(x[:, :, :3]).to(x.device)
        self._update_graph()
        
        x = F.relu(self.remap(x))
        assert (x.size(1) == self.max_size)

        s = x.size()
        x = x.view(-1, s[2])

        x = F.relu(self.conv1(x) + x)
        x = F.relu(self.conv2(x) + x)
        x = F.relu(self.conv3(x) + x)
        x = self.conv4(x)

        x = x.view(s[0], s[1], -1)

        return x


if __name__ == "__main__":
    data = np.load(osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives/data/push/numpy_push_cuboid_1/numpy_push_cuboid_1/0_102_1.npz'))
    pcd = data['start'][:100]

    knn = KNNGraph(pcd.shape[0])
    g = knn(torch.from_numpy(pcd).float()).to('cuda:0')

    h = torch.from_numpy(pcd).float()
    h = torch.cat((h - torch.mean(h, dim=0), torch.mean(h, dim=0).repeat(h.size(0), 1)), dim=1).to('cuda:0')

    model = GeomEncoder(6, 256, 1).cuda()
    out = model(h[None, :, :])
    print(out)

    from IPython import embed
    embed()

    