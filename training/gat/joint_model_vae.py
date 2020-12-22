import argparse
import torch
import copy
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
# from models_pointnet import PointNet
from quantizer import VectorQuantizer
# from gat_dgl import GeomEncoder as GeomEncoderDGL
# from gat_pyg import GeomEncoder as GeomEncoderPyG

class Encoder(nn.Module):

    def __init__(self,
                 in_dim,
                 mu_out_dim,
                 logvar_out_dim,
                 hidden_layers):
        super(Encoder, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.in_fc = nn.Linear(in_dim, hidden_layers[0])

        for i in range(len(hidden_layers) - 1):
            fc = nn.Linear(hidden_layers[i], hidden_layers[i+1])
            self.hidden_layers.append(fc)

        self.out_mu_head = nn.Linear(hidden_layers[-1], mu_out_dim)
        self.out_logvar_head = nn.Linear(hidden_layers[-1], logvar_out_dim)

    def forward(self, x):
        h = F.relu(self.in_fc(x))
        for i, layer in enumerate(self.hidden_layers):
            h = F.relu(layer(h))
        mu_out = self.out_mu_head(h)
        logvar_out = self.out_logvar_head(h)
        return mu_out, logvar_out


class Decoder(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_layers):
        super(Decoder, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.in_fc = nn.Linear(in_dim, hidden_layers[0])

        for i in range(len(hidden_layers) - 1):
            fc = nn.Linear(hidden_layers[i], hidden_layers[i+1])
            self.hidden_layers.append(fc)

        self.out_right_head = nn.Linear(hidden_layers[-1], out_dim)
        self.out_left_head = nn.Linear(hidden_layers[-1], out_dim)

    def forward(self, x):
        h = F.relu(self.in_fc(x))
        for i, layer in enumerate(self.hidden_layers):
            h = F.relu(self.hidden_layers[i](h))
        return self.out_right_head(h), self.out_left_head(h)


class JointVAEFull(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim,
                 decoder_input_dim, hidden_layers, pyg=True):
        super(JointVAEFull, self).__init__()
        encoder_hidden_layers = copy.deepcopy(hidden_layers)
        decoder_hidden_layers = copy.deepcopy(hidden_layers)
        decoder_hidden_layers.reverse()
        
        if pyg:
            # GeomEncoder = GeomEncoderPyG
            from gat_pyg import GeomEncoder
        else:
            # GeomEncoder = GeomEncoderDGL
            from gat_dgl import GeomEncoder

        self.palm_decoder = Decoder(
            in_dim=latent_dim,
            out_dim=out_dim,
            hidden_layers=decoder_hidden_layers)

        # points: [x, y, z, com_x, com_y, com_z]
        p_dim, mask_dim, transform_dim, trans_dim, palms_dim = 6, 1, 7, 2, 14
        self.geom_encoder = GeomEncoder(
            in_dim=p_dim + mask_dim + transform_dim + trans_dim + palms_dim,
            latent_dim=latent_dim,
            )        

        self.geom_decoder = GeomEncoder(
            in_dim=2*latent_dim,
            latent_dim=latent_dim,
            )

        self.embed_geom = nn.Linear(6, latent_dim)

        self.output_mu_head = nn.Linear(latent_dim, latent_dim)
        self.output_logvar_head = nn.Linear(latent_dim, latent_dim)
        self.latent_dim = latent_dim

        self.exist_wt = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())
        self.mask_head = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())
        self.translation = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 2))
        self.transformation = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 7))

        self.bce_loss = torch.nn.BCELoss()
        self.mse = nn.MSELoss(reduction='mean')
        self.beta = 0.01

    def encode(self, x):
        output_geom = F.relu(self.geom_encoder(x))
        output_geom = output_geom.mean(dim=1)
        z_mu = self.output_mu_head(output_geom)
        z_logvar = self.output_logvar_head(output_geom)
        z = self.reparameterize(z_mu, z_logvar)
        return z, z_mu, z_logvar

    def decode(self, z, decoder_x, vq=False, *args, **kwargs):
        geom_embed = self.embed_geom(decoder_x)
        # ex_wt = self.exist_wt(geom_embed)
 
        if len(z.size()) == 3:
            pass
        else:
            z = z[:, None, :].repeat(1, geom_embed.size(1), 1)

        full_geom_embed = torch.cat((z, geom_embed), dim=-1)
        latent = self.geom_decoder(full_geom_embed, full=True)
        pred_mask = self.mask_head(latent)
        ex_wt = self.exist_wt(latent)
        output_r, output_l = self.palm_decoder(latent)

        latent = latent.mean(dim=1)

        translation = self.translation(latent)
        transformation = self.transformation(latent)
        recon_mu = (output_r, output_l, pred_mask, translation, transformation)

        return recon_mu, ex_wt

    def forward(self, x, decoder_x, vq=False, *args, **kwargs):
        z, z_mu, z_logvar = self.encode(x)
        recon_mu, ex_wt = self.decode(z, decoder_x, vq=vq)

        return z, recon_mu, z_mu, z_logvar, ex_wt

    def reparameterize(self, mu, logvar):
        std_dev = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std_dev)
        z = mu + std_dev * eps
        return z

    def kl_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
        kl_loss_mean = torch.mean(kl_loss)
        return kl_loss_mean

    def existence_loss(self, prediction, target):
        bce_loss = self.bce_loss(prediction, target)
        return bce_loss

    def translation_loss(self, prediction, target):
        pred_loss = torch.pow(prediction - target, 2).mean()
        return pred_loss

    def normal_vec_loss(self, prediction, target):
        prediction_norms = torch.norm(prediction, p=2, dim=1)
        normalized_prediction = torch.cuda.FloatTensor(prediction.shape).fill_(0)

        for i in range(normalized_prediction.shape[0]):
            normalized_prediction[i, :] = prediction[i, :]/prediction_norms[i]

        norm_loss = torch.mean(self.beta*torch.pow((1 - torch.norm(prediction, p=2, dim=1)), 2))
        position_loss = self.mse(normalized_prediction, target) + norm_loss
        return position_loss

    def rotation_loss(self, prediction, target):
        normalized_prediction = F.normalize(prediction, p=2, dim=1)
        scalar_prod = torch.sum(normalized_prediction * target, axis=1)
        dist = 1 - torch.pow(scalar_prod, 2)

        norm_loss = self.beta*torch.pow((1 - torch.norm(prediction, p=2, dim=1)), 2)
        rotation_loss = dist + norm_loss
        mean_rotation_loss = torch.mean(rotation_loss)
        return mean_rotation_loss


class JointPointVAE(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim,
                 decoder_input_dim, hidden_layers, pyg=True):
        super(JointPointVAE, self).__init__()
        encoder_hidden_layers = copy.deepcopy(hidden_layers)
        decoder_hidden_layers = copy.deepcopy(hidden_layers)
        decoder_hidden_layers.reverse()
        
        if pyg:
            from models_pointnet import PointNet
        else:
            raise ValueError('Non PyTorch Geometric PointNet not yet implemented')

        self.palm_decoder = Decoder(
            in_dim=latent_dim,
            out_dim=out_dim,
            hidden_layers=decoder_hidden_layers)

        p_dim, palm_dim, trans_dim, mask_dim = 6, 14, 7, 1
        self.geom_encoder = PointNet(
            in_dim=p_dim+palm_dim+trans_dim+mask_dim,
            latent_dim=latent_dim,
            )        

        self.geom_decoder = PointNet(
            in_dim=6 + latent_dim,
            latent_dim=latent_dim,
            )

        self.embed_geom = nn.Linear(6, latent_dim)

        self.output_mu_head = nn.Linear(latent_dim, latent_dim)
        self.output_logvar_head = nn.Linear(latent_dim, latent_dim)
        self.latent_dim = latent_dim

        self.exist_wt = nn.Sequential(nn.Linear(latent_dim+6, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())
        self.mask_head = nn.Sequential(nn.Linear(latent_dim+6, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())
        self.transformation = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 7))

        self.bce_loss = torch.nn.BCELoss()
        self.mse = nn.MSELoss(reduction='mean')
        self.beta = 0.01

    def encode(self, x):
        s = x.size()
        nrep = s[1]
        batch = torch.arange(0, s[0])
        batch = batch[:, None].repeat(1, nrep)
        batch = batch.view(-1).long().to(x.device)

        x = x.view(-1, s[2])
        output_geom = self.geom_encoder(x[:, 3:], x[:, :3], batch)        
        z_mu = self.output_mu_head(output_geom)
        z_logvar = self.output_logvar_head(output_geom)
        z = self.reparameterize(z_mu, z_logvar)
        return z, z_mu, z_logvar

    def decode(self, z, decoder_x, vq=False, *args, **kwargs):
        if vq:
            loss, z = self.quantizer(z)
        else:
            loss = None        
        z = z[:, None, :].repeat(1, decoder_x.size(1), 1)

        full_geom_embed = torch.cat((decoder_x, z), dim=-1)

        s = full_geom_embed.size()
        nrep = s[1]
        batch = torch.arange(0, s[0])
        batch = batch[:, None].repeat(1, nrep)
        batch = batch.view(-1).long().to(full_geom_embed.device)

        full_geom_embed = full_geom_embed.view(-1, s[2])

        latent = self.geom_decoder(full_geom_embed[:, 3:], full_geom_embed[:, :3], batch)
        x_latent_cat = torch.cat((decoder_x, latent[:, None, :].repeat(1, decoder_x.size(1), 1)), dim=2)
        pred_mask = self.mask_head(x_latent_cat)
        output_r, output_l = self.palm_decoder(latent)
        # output_r, output_l = self.palm_decoder(x_latent_cat)

        transformation = self.transformation(latent)
        recon_mu = (output_r, output_l, pred_mask, transformation[:, :2], transformation)
        return recon_mu, None, loss

    def forward(self, x, decoder_x, vq=False, *args, **kwargs):
        z, z_mu, z_logvar = self.encode(x)
        recon_mu, ex_wt, loss = self.decode(z, decoder_x, vq=vq)

        return z, recon_mu, z_mu, z_logvar, ex_wt, loss

    def reparameterize(self, mu, logvar):
        std_dev = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std_dev)
        z = mu + std_dev * eps
        return z

    def kl_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
        kl_loss_mean = torch.mean(kl_loss)
        return kl_loss_mean

    def existence_loss(self, prediction, target):
        bce_loss = self.bce_loss(prediction, target)
        return bce_loss

    def translation_loss(self, prediction, target):
        pred_loss = torch.pow(prediction - target, 2).mean()
        return pred_loss

    def normal_vec_loss(self, prediction, target):
        prediction_norms = torch.norm(prediction, p=2, dim=1)
        normalized_prediction = torch.cuda.FloatTensor(prediction.shape).fill_(0)

        for i in range(normalized_prediction.shape[0]):
            normalized_prediction[i, :] = prediction[i, :]/prediction_norms[i]

        norm_loss = torch.mean(self.beta*torch.pow((1 - torch.norm(prediction, p=2, dim=1)), 2))
        position_loss = self.mse(normalized_prediction, target) + norm_loss
        return position_loss

    def rotation_loss(self, prediction, target):
        normalized_prediction = F.normalize(prediction, p=2, dim=1)

        scalar_prod = torch.sum(normalized_prediction * target, axis=1)
        dist = 1 - torch.pow(scalar_prod, 2)

        norm_loss = self.beta*torch.pow((1 - torch.norm(prediction, p=2, dim=1)), 2)
        rotation_loss = dist + norm_loss
        mean_rotation_loss = torch.mean(rotation_loss)
        return mean_rotation_loss
