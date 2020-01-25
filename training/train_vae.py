import os
import argparse
import time
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from data_loader import DataLoader
from model import VAE
from util import to_var, save_state, load_net_state, load_seed, load_opt_state


def main(args):
    batch_size = args.batch_size
    dataset_size = args.total_data_size

    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)

    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)

    trained_model_path = os.path.join(args.model_path, args.model_name)
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)

    vae = VAE(
        args.input_dimension,
        args.output_dimension,
        args.latent_dimension,
        lr=args.learning_rate
    )

    if torch.cuda.is_available():
        vae.encoder.cuda()
        vae.decoder.cuda()

    if args.start_epoch > 0:
        fname = os.path.join(
            trained_model_path,
            args.model_name+'_epoch_%d.pt' % args.start_epoch)
        torch_seed, np_seed = load_seed(fname)
        load_net_state(vae)
        load_opt_state(vae)
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)

    data_dir = args.data_dir
    data_loader = DataLoader(data_dir=data_dir)

    data_loader.create_random_ordering(size=dataset_size)

    total_loss = []
    start_time = time.time()
    for epoch in range(args.start_epoch, args.start_epoch+args.num_epochs):
        print('Epoch: ' + str(epoch))
        epoch_total_loss = 0
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

            epoch_total_loss = epoch_total_loss + loss.data
        print(' --avgerage loss: ')
        print(epoch_total_loss/(dataset_size/batch_size))
        total_loss.append(epoch_total_loss/(dataset_size/batch_size))

        if epoch % args.save_freq == 0:
            print('\n--Saving model\n')
            print('time: ' + str(time.time() - start_time))

            save_state(
                net=vae,
                torch_seed=torch_seed,
                np_seed=np_seed,
                fname=os.path.join(
                    trained_model_path,
                    args.model_name+'_epoch_'+str(epoch) + '.pt'))

            np.savez(
                os.path.join(
                    trained_model_path,
                    args.model_name+'_epoch_'+str(epoch) + '_loss.npz'),
                loss=np.asarray(total_loss))

    print('Done!')
    save_state(
        net=vae,
        torch_seed=torch_seed,
        np_seed=np_seed,
        fname=os.path.join(
            trained_model_path,
            args.model_name+'_epoch_'+str(epoch) + '.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int,
                        default=200)
    parser.add_argument('--num_epochs', type=int,
                        default=50)
    parser.add_argument('--total_data_size', type=int,
                        default=6000)
    parser.add_argument('--data_dir', type=str,
                        default='/root/catkin_ws/src/primitives/data/pull/face_ind_large_0_fixed')
    parser.add_argument('--input_dimension', type=int,
                        default=14)
    parser.add_argument('--latent_dimension', type=int,
                        default=3)
    parser.add_argument('--output_dimension', type=int,
                        default=7)
    parser.add_argument('--learning_rate', type=float,
                        default=3e-3)
    parser.add_argument('--model_path', type=str,
                        default='/root/training/saved_models')
    parser.add_argument('--start_epoch', type=int,
                        default=0)
    parser.add_argument('--args.save_freq', type=int,
                        default=2)
    parser.add_argument('--model_name', type=str)

    args = parser.parse_args()
    main(args)