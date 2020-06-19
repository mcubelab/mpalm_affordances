from torch.nn import ModuleList
import torch.nn.functional as F
import torch.nn as nn
import torch
from easydict import EasyDict
from torch.nn.utils import spectral_norm



class EnergyModel(nn.Module):
    def __init__(self):
        super(EnergyModel, self).__init__()
        self.fc1 = nn.Linear(28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 128)
        self.energy_map = spectral_norm(nn.Linear(128, 1))


    def forward(self, x):
        # print(x.size())
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        energy = self.energy_map(F.normalize(h, p=2) / 0.07)

        return energy
