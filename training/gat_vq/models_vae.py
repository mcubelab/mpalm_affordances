import argparse
import torch
import copy
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch_geometric.nn import GATConv
from quantizer import VectorQuantizer
import numpy as np


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


class GoalDecoder(Decoder):
    def __init__(self, in_dim, out_dim, hidden_layers):
        super(GoalDecoder, self).__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_layers=hidden_layers)
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_layers[-1], out_dim)
        )

    def forward(self, x):
        h = F.relu(self.in_fc(x))
        for i, layer in enumerate(self.hidden_layers):
            h = F.relu(self.hidden_layers[i](h))
        return self.goal_head(h)


class VAE(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim,
                 decoder_input_dim, hidden_layers):
        super(VAE, self).__init__()
        encoder_hidden_layers = copy.deepcopy(hidden_layers)
        decoder_hidden_layers = copy.deepcopy(hidden_layers)
        decoder_hidden_layers.reverse()
        self.encoder = Encoder(
            in_dim=in_dim,
            mu_out_dim=latent_dim,
            logvar_out_dim=latent_dim,
            hidden_layers=encoder_hidden_layers)
        self.decoder = Decoder(
            in_dim=latent_dim+decoder_input_dim,
            out_dim=out_dim,
            hidden_layers=decoder_hidden_layers)
        self.mse = nn.MSELoss(reduction='mean')
        self.beta = 1.0

    def encode(self, x):
        z_mu, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mu, z_logvar)
        return z, z_mu, z_logvar

    def decode(self, z, decoder_x):
        if not torch.isnan(decoder_x).any():
            inp = torch.cat((z, decoder_x), dim=-1)
        else:
            inp = z
        recon_mu = self.decoder(inp)
        return recon_mu

    def forward(self, x, decoder_x):
        z, z_mu, z_logvar = self.encode(x)
        recon_mu = self.decode(z, decoder_x)
        ex_wt = F.sigmoid(self.existence_prob(z))

        return z, recon_mu, z_mu, z_logvar

    def reparameterize(self, mu, logvar):
        std_dev = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std_dev)
        z = mu + std_dev * eps
        return z

    def kl_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
        kl_loss_mean = torch.mean(kl_loss)
        return kl_loss_mean

    def rotation_loss(self, prediction, target):
        # prediction_norms = torch.norm(prediction, p=2, dim=1)
        # normalized_prediction = torch.cuda.FloatTensor(prediction.shape).fill_(0)

        # for i in range(normalized_prediction.shape[0]):
        #     normalized_prediction[i, :] = prediction[i, :]/prediction_norms[i]
        normalized_prediction = F.normalize(prediction, p=2, dim=-1)

        scalar_prod = torch.sum(normalized_prediction * target, axis=1)
        dist = 1 - torch.pow(scalar_prod, 2)

        norm_loss = self.beta*torch.pow((1 - torch.norm(prediction, p=2, dim=1)), 2)
        rotation_loss = dist + norm_loss
        mean_rotation_loss = torch.mean(rotation_loss)
        return mean_rotation_loss

    def recon_loss(self, prediction, target):
        pos_prediction = prediction[:, :3]
        pos_target = target[:, :3]
        pos_loss = self.mse(pos_prediction, pos_target)

        ori_prediction = prediction[:, 3:]
        ori_target = target[:, 3:]
        ori_loss = self.rotation_loss(ori_prediction, ori_target)
        return pos_loss + ori_loss

    def total_loss(self, prediction, target, mu, logvar):
        kl_loss = self.kl_loss(mu, logvar)
        recon_loss = self.recon_loss(prediction, target)
        total_loss = kl_loss + recon_loss
        return total_loss


class GeomVAE(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim,
                 decoder_input_dim, hidden_layers, table_mesh=False):
        super(GeomVAE, self).__init__()
        encoder_hidden_layers = copy.deepcopy(hidden_layers)
        decoder_hidden_layers = copy.deepcopy(hidden_layers)
        decoder_hidden_layers.reverse()

        self.encoder = Encoder(
            in_dim=in_dim,
            mu_out_dim=latent_dim,
            logvar_out_dim=latent_dim,
            hidden_layers=hidden_layers
            )

        self.geom_encoder = GeomEncoder(
            in_dim=7,
            latent_dim=latent_dim,
            table_mesh=table_mesh,
            )

        self.decoder = Decoder(
            in_dim=2*latent_dim,
            out_dim=out_dim,
            hidden_layers=decoder_hidden_layers)

        self.exist_wt = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())
        self.mse = nn.MSELoss(reduction='mean')
        self.beta = 1.0

    def encode(self, x):
        z_mu, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mu, z_logvar)
        return z, z_mu, z_logvar

    def decode(self, z, decoder_x, select_idx, full=False):
        geom_embed = self.geom_encoder(decoder_x)

        if full:
            repeat = select_idx.size(1)
        else:
            repeat = 20

        ex_wt = self.exist_wt(geom_embed)
        select_idx = select_idx[:, :, None].repeat(1, 1, geom_embed.size(2))[:, :repeat]
        geom_embed = torch.gather(geom_embed, 1, select_idx)

        size = geom_embed.size()
        z = z[:, None, :].repeat(1, size[1], 1)

        inp = torch.cat((z * geom_embed, geom_embed), dim=-1)
        recon_mu = self.decoder(inp)

        return recon_mu, ex_wt

    def forward(self, x, decoder_x, select_idx, full=False):

        z, z_mu, z_logvar = self.encode(x)
        recon_mu, ex_wt = self.decode(z, decoder_x, select_idx, full=full)

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

    def rotation_loss(self, prediction, target):
        prediction_norms = torch.norm(prediction, p=2, dim=1)
        normalized_prediction = torch.cuda.FloatTensor(prediction.shape).fill_(0)

        for i in range(normalized_prediction.shape[0]):
            normalized_prediction[i, :] = prediction[i, :]/prediction_norms[i]

        scalar_prod = torch.sum(normalized_prediction * target, axis=1)
        dist = 1 - torch.pow(scalar_prod, 2)

        norm_loss = self.beta*torch.pow((1 - torch.norm(prediction, p=2, dim=1)), 2)
        rotation_loss = dist + norm_loss
        mean_rotation_loss = torch.mean(rotation_loss)
        return mean_rotation_loss

    def recon_loss(self, prediction, target):
        pos_prediction = prediction[:, :3]
        pos_target = target[:, :3]
        pos_loss = self.mse(pos_prediction, pos_target)

        ori_prediction = prediction[:, 3:]
        ori_target = target[:, 3:]
        ori_loss = self.rotation_loss(ori_prediction, ori_target)
        return pos_loss + ori_loss

    def total_loss(self, prediction, target, mu, logvar):
        kl_loss = self.kl_loss(mu, logvar)
        recon_loss = self.recon_loss(prediction, target)
        total_loss = kl_loss + recon_loss
        return total_loss


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


class GoalVAE(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim,
                 decoder_input_dim, hidden_layers):
        super(GoalVAE, self).__init__()
        encoder_hidden_layers = copy.deepcopy(hidden_layers)
        decoder_hidden_layers = copy.deepcopy(hidden_layers)
        decoder_hidden_layers.reverse()

        self.geom_encoder = GeomEncoder(
            in_dim=9,
            latent_dim=latent_dim,
            )

        self.output_mu_head = nn.Linear(latent_dim, latent_dim)
        self.output_logvar_head = nn.Linear(latent_dim, latent_dim)
        self.latent_dim = latent_dim

        self.embed_geom = nn.Linear(6, latent_dim)

        self.geom_decoder = GeomEncoder(
            in_dim=2*latent_dim,
            latent_dim=latent_dim,
            )

        self.exist_wt = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())
        self.translation = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 2))
        self.bce_loss = torch.nn.BCELoss()
        self.quantizer = VectorQuantizer(128, latent_dim, 0.25)

    def encode(self, x):
        output_geom = F.relu(self.geom_encoder(x))
        output_geom = output_geom.mean(dim=1)
        z_mu = self.output_mu_head(output_geom)
        z_logvar = self.output_logvar_head(output_geom)
        z = self.reparameterize(z_mu, z_logvar)
        return z, z_mu, z_logvar

    def decode(self, z, decoder_x):
        loss, z = self.quantizer(z)

        geom_embed = self.embed_geom(decoder_x)
        z = z[:, None, :].repeat(1, geom_embed.size(1), 1)

        combined_geom = torch.cat([geom_embed, z], dim=2)
        latent = self.geom_decoder(combined_geom, full=True)
        pred_mask = self.exist_wt(latent)
        latent = latent.mean(dim=1)
        translation = self.translation(latent)

        return pred_mask, translation, loss

    def forward(self, x, decoder_x):

        z, z_mu, z_logvar = self.encode(x)
        recon_mu = self.decode(z, decoder_x)

        return z, recon_mu, z_mu, z_logvar

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

def main(args):
    batch_size = args.batch_size
    dataset_size = args.total_data_size

    cfg = get_vae_defaults()
    # cfg.merge_from_file(cfg_file)
    cfg.freeze()

    vae = VAE(
        args.input_dimension,
        args.output_dimension,
        args.latent_dimension,
        hidden_layers=cfg.ENCODER_HIDDEN_LAYERS_MLP
    )

    if torch.cuda.is_available():
        vae.encoder.cuda()
        vae.decoder.cuda()

    data_dir = args.data_dir
    data_loader = DataLoader(data_dir=data_dir)

    data_loader.create_random_ordering(size=dataset_size)

    total_loss = 0
    for epoch in range(args.num_epochs):
        print('Epoch: ' + str(epoch))
        running_total_loss = 0
        for i in range(0, dataset_size, batch_size):
            vae.optimizer.zero_grad()

            input_batch, target_batch = data_loader.load_batch(i, batch_size)
            input_batch = to_var(torch.from_numpy(input_batch))
            target_batch = to_var(torch.from_numpy(target_batch))

            z, recon_mu, z_mu, z_logvar = vae.forward(input_batch)
            # target = torch.normal(z_mu)
            output = recon_mu

            kl_loss = vae.kl_loss(z_mu, z_logvar)
            recon_loss = vae.recon_loss(output, target_batch)
            # loss = vae.total_loss(output, target_batch, z_mu, z_logvar)
            loss = kl_loss + recon_loss
            loss.backward()
            vae.optimizer.step()

            running_total_loss = running_total_loss + loss.data
            print(" -- running loss: ")
            print(running_total_loss)
        print(' --avgerage loss: ')
        print(running_total_loss/(dataset_size/batch_size))
        total_loss = total_loss + running_total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int,
                        default=32)
    parser.add_argument('--num_epochs', type=int,
                        default=1)
    parser.add_argument('--total_data_size', type=int,
                        default=1024)
    parser.add_argument('--data_dir', type=str,
                        default='/home/anthony/repos/research/mpalm_affordances/catkin_ws/src/primitives/data/pull/face_ind_large_0_fixed')
    parser.add_argument('--input_dimension', type=int,
                        default=14)
    parser.add_argument('--latent_dimension', type=int,
                        default=6)
    parser.add_argument('--output_dimension', type=int,
                        default=7)

    args = parser.parse_args()
    main(args)
