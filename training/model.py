import argparse
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from data_loader import DataLoader
from vae_cfg import get_vae_defaults


class Encoder(nn.Module):

    def __init__(self,
                 in_dim,
                 mu_out_dim,
                 logvar_out_dim,
                 hidden_layers):
        super(Encoder, self).__init__()

        # self.hidden_layers = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim), F.relu(),
        #     nn.Linear(hidden_dim, hidden_dim), F.relu()
        # )

        # self.hidden_layers = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim), F.relu(),
        # )
        # self.mu_head = nn.Sequential(
        #     nn.Linear(hidden_dim, mu_out_dim)
        # )
        # self.logvar_head = nn.Sequential(
        #     nn.Linear(hidden_dim, logvar_out_dim)
        # )

        self.hidden_layers = []
        self.in_fc = nn.Linear(in_dim, hidden_layers[0])

        for i in range(len(hidden_layers) - 1):
            fc = nn.Linear(hidden_layers[i], hidden_layers[i+1])
            self.hidden_layers.append(fc)

        self.out_mu_head = nn.Linear(hidden_layers[-1], mu_out_dim)
        self.out_logvar_head = nn.Linear(hidden_layers[-1], logvar_out_dim)

        if torch.cuda.is_available():
            for layer in self.hidden_layers:
                layer.cuda()  # why is this necessary?
            
    def forward(self, x):
        # h = self.hidden_layers(x)
        # mu_out = self.mu_head(h)
        # logvar_out = self.logvar_head(h)

        h = F.relu(self.in_fc(x))
        for i, layer in enumerate(self.hidden_layers):
            h = F.relu(layer(h))
        mu_out = self.out_mu_head(h)
        logvar_out = self.out_logvar_head(h)
        return mu_out, logvar_out


class Decoder(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_layers):
        super(Decoder, self).__init__()
        # self.decoder = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim), F.relu(),
        #     nn.Linear(hidden_dim, hidden_dim), F.relu(),
        #     nn.Linear(hidden_dim, hidden_dim), F.relu()
        # )
        # self.right_head = nn.Sequential(
        #     nn.Linear(hidden_dim, out_dim)
        # )
        # self.left_head = nn.Sequential(
        #     nn.Linear(hidden_dim, out_dim)
        # )
        self.hidden_layers = []
        self.in_fc = nn.Linear(in_dim, hidden_layers[0])

        for i in range(len(hidden_layers) - 1):
            fc = nn.Linear(hidden_layers[i], hidden_layers[i+1])
            self.hidden_layers.append(fc)

        self.out_right_head = nn.Linear(hidden_layers[-1], out_dim)
        self.out_left_head = nn.Linear(hidden_layers[-1], out_dim)

        if torch.cuda.is_available():
            for layer in self.hidden_layers:
                layer.cuda()  # why is this necessary?        

    def forward(self, x):
        # h = self.decoder(x)
        h = F.relu(self.in_fc(x))
        for i, layer in enumerate(self.hidden_layers):
            h = F.relu(self.hidden_layers[i](h))
        return self.right_head(h), self.left_head(h)


class GoalDecoder(Decoder):
    # def __init__(self, z_dim, start_dim, out_palm_dim, out_goal_dim, hidden_dim=128):
    def __init__(self, in_dim, out_dim, hidden_layers):
        super(GoalDecoder, self).__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_layers=hidden_layers)
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_layers[-1], out_dim)
        )

    def forward(self, x):
        # h = self.decoder(x)
        h = F.relu(self.in_fc(x))
        for i, layer in enumerate(self.hidden_layers):
            h = F.relu(self.hidden_layers[i](h))
        return self.goal_head(h)

    # def forward(self, z, start, goal=None):
    #     """
    #     Full forward pass on dual head goal state + palm_pose decoder

    #     Args:
    #         z (FloatTensor): Sampled value from encoder latent space
    #         start (FloatTensor): Start state/observation representation
    #         of object

    #     Returns:
    #         3-element tuple containing:
    #         - FloatTensor: Reconstruction of goal
    #         - FloatTensor: Reconstruction of right_palm pose pose
    #         - FloatTensor: Reconstruction of left_palm pose
    #     """
    #     x_goal = torch.cat((z, start), dim=-1)
    #     g_hat = self.goal_forward(x_goal)
    #     if goal is None:
    #         x_palms = torch.cat((x_goal, g_hat), dim=-1)  # use reconstructed goal
    #     else:
    #         x_palms = torch.cat((x_goal, goal), dim=-1)  # use ground truth goal
    #     right_palm, left_palm = self.palms_forward(x_palms)
    #     return g_hat, right_palm, left_palm

    # def palms_forward(self, x):
    #     """
    #     Output a reconstruction of the palm poses, given
    #     the start and goal concatenated with a sampled value
    #     from the latent space of the encoder

    #     Args:
    #         x (FloatTensor): Input to the palm head of the model,
    #             concatenated variable with [z, start, goal]

    #     Returns:
    #         FloatTensor: Palm pose reconstruction [right, left] \in SE(3),
    #             in the object frame
    #     """
    #     h = self.decoder(x)
    #     return self.right_head(h), self.left_head(h)

    # def goal_forward(self, x):
    #     """
    #     Output a reconstruction of the object goal state, given the
    #     start state concatenate with a sampled value from the latent
    #     space of the encoder

    #     Args:
    #         x (FloatTensor): Input to the goal head of the model,
    #             concatenated variable with [z, start]

    #     Returns:
    #         FloatTensor: Goal representation reconstruction
    #     """
    #     h = self.goal_decoder(x)
    #     return self.goal_head(h)


class VAE(object):
    def __init__(self, in_dim, out_dim, latent_dim, 
                 decoder_input_dim, hidden_layers, lr):
        # decoder_input_dim = in_dim - out_dim*2
        self.encoder = Encoder(
            in_dim=in_dim,
            mu_out_dim=latent_dim,
            logvar_out_dim=latent_dim,
            hidden_layers=hidden_layers)
        self.decoder = Decoder(
            in_dim=latent_dim+decoder_input_dim,
            out_dim=out_dim,
            hidden_layers=hidden_layers)
        self.mse = nn.MSELoss(reduction='mean')

        params = list(self.encoder.parameters()) +  \
            list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        self.beta = 0.001

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


class GoalVAE(VAE):
    def __init__(self, in_dim, out_dim, latent_dim, 
                 decoder_input_dim, hidden_layers, lr):
        # decoder_input_dim = in_dim - out_dim*2
        super(GoalVAE, self).__init__(
            in_dim, out_dim, latent_dim, 
            decoder_input_dim, hidden_layers, lr
        )
        self.decoder = GoalDecoder(
            in_dim=latent_dim+decoder_input_dim,
            out_dim=out_dim,
            hidden_layers=hidden_layers
        )

        params = list(self.encoder.parameters()) +  \
            list(self.decoder.parameters())

        self.optimizer = optim.Adam(params, lr=lr)

    #def decode(self, z):
    #    recon_mu = self.decoder(z)
    #    return recon_mu

    #def forward(self, x):
    #    z, z_mu, z_logvar = self.encode(x)
    #    recon_mu = self.decode(z)
    #    return z, recon_mu, z_mu, z_logvar


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

    # cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
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
