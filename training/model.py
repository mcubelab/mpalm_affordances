import argparse
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from data_loader import DataLoader


class Encoder(nn.Module):

    def __init__(self,
                 in_dim,
                 mu_out_dim,
                 logvar_out_dim,
                 mu_head_in,
                 logvar_head_in,
                 hidden_dim=64):
        super(Encoder, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, mu_out_dim)
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, logvar_out_dim)
        )

    def forward(self, x):
        h = self.hidden_layers(x)
        mu_out = self.mu_head(h)
        logvar_out = self.logvar_head(h)
        return mu_out, logvar_out


class Decoder(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class VAE():
    def __init__(self, in_dim, out_dim, latent_dim, lr):
        self.encoder = Encoder(
            in_dim=in_dim,
            mu_out_dim=latent_dim,
            logvar_out_dim=latent_dim,
            mu_head_in=64,
            logvar_head_in=64)
        self.decoder = Decoder(
            in_dim=latent_dim,
            out_dim=out_dim)
        self.mse = nn.MSELoss()

        params = list(self.encoder.parameters()) + \
            list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        self.beta = 0

    def encode(self, x):
        z_mu, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mu, z_logvar)
        return z, z_mu, z_logvar

    def decode(self, z):
        recon_mu = self.decoder(z)
        return recon_mu

    def forward(self, x):
        z, z_mu, z_logvar = self.encode(x)
        recon_mu = self.decode(z)
        return z, recon_mu, z_mu, z_logvar

    def reparameterize(self, mu, logvar):
        std_dev = torch.exp((0.5) * logvar)
        eps = torch.normal(mu, std_dev)
        z = mu + std_dev * eps
        return z

    def kl_loss(self, mu, sigma):
        kl_loss = 0.5 * torch.sum(1 + torch.log(torch.pow(sigma, 2)) -
                                  torch.pow(sigma, 2) - torch.pow(mu, 2))
        kl_loss_mean = torch.mean(kl_loss)
        return kl_loss_mean

    def rotation_loss(self, prediction, target):
        # scalar_prod = torch.mm(prediction, target)
        scalar_prod = torch.sum(prediction*target, axis=1)
        # angle = 2*torch.pow(scalar_prod, 2) - 1
        # angle = 2*torch.acos(scalar_prod)
        dist = 1 - torch.pow(scalar_prod, 2)
        norm_loss = self.beta*torch.pow((1 - torch.norm(prediction)), 2)
        rotation_loss = dist * norm_loss
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
        sigma = torch.exp(0.5 * logvar)
        kl_loss = self.kl_loss(mu, sigma)
        recon_loss = self.recon_loss(prediction, target)
        total_loss = kl_loss + recon_loss
        return total_loss


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def main(args):
    # reconstruction loss
    # (1 / L) * torch.sum(torch.log())
    # mse?
    batch_size = args.batch_size
    dataset_size = args.total_data_size

    vae = VAE(
        args.input_dimension,
        args.output_dimension,
        args.latent_dimension
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
            vae.encoder.zero_grad()
            vae.decoder.zero_grad()

            input_batch, target_batch = data_loader.load_batch(i, batch_size)
            input_batch = to_var(torch.from_numpy(input_batch))
            target_batch = to_var(torch.from_numpy(target_batch))

            z, recon_mu, z_mu, z_logvar = vae.forward(input_batch)
            # target = torch.normal(z_mu)
            output = recon_mu

            loss = vae.total_loss(output, target_batch, z_mu, z_logvar)
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
                        default=100)
    parser.add_argument('--num_epochs', type=int,
                        default=1)
    parser.add_argument('--total_data_size', type=int,
                        default=1000)
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
