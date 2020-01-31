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

    # torch_seed = np.random.randint(low=0, high=1000)
    # np_seed = np.random.randint(low=0, high=1000)
    torch_seed = 0
    np_seed = 0

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
        load_net_state(vae, fname)
        load_opt_state(vae, fname)
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)

    data_dir = args.data_dir
    data_loader = DataLoader(data_dir=data_dir)

    data_loader.create_random_ordering(size=dataset_size)

    total_loss = []
    running_kl_loss = []
    running_recon_loss = []
    running_ori_loss = []
    running_pos_loss = []

    running_ori_outputs = []
    running_pos_outputs = []
    start_time = time.time()
    for epoch in range(args.start_epoch, args.start_epoch+args.num_epochs):
        print('Epoch: ' + str(epoch))
        epoch_total_loss = 0
        epoch_kl_loss = 0
        epoch_pos_loss = 0
        epoch_ori_loss = 0
        epoch_recon_loss = 0
        for i in range(0, dataset_size, batch_size):
            vae.optimizer.zero_grad()

            input_batch, target_batch = data_loader.load_batch(i, batch_size, start_rep=args.start_rep)
            input_batch = to_var(torch.from_numpy(input_batch))

            decoder_input = input_batch[:, ]

            # z, recon_mu, z_mu, z_logvar = vae.forward(input_batch)
            # target = torch.normal(z_mu)
            # output = recon_mu

            # kl_loss = vae.kl_loss(z_mu, z_logvar)
            if args.start_rep == 'keypoints':

                decoder_input = input_batch[:, :31]
                z, recon_mu, z_mu, z_logvar = vae.forward(input_batch, decoder_input)
                kl_loss = vae.kl_loss(z_mu, z_logvar)
        
                target_batch_right = to_var(torch.from_numpy(target_batch[:, 0]))
                target_batch_left = to_var(torch.from_numpy(target_batch[:, 1]))

                output_r, output_l = recon_mu

                pos_loss_right = vae.mse(
                    output_r[:, :3],
                    target_batch_right[:, :3])
                ori_loss_right = vae.rotation_loss(
                    output_r[:, 3:],
                    target_batch_right[:, 3:])

                pos_loss_left = vae.mse(
                    output_l[:, :3],
                    target_batch_left[:, :3])
                ori_loss_left = vae.rotation_loss(
                    output_l[:, 3:],
                    target_batch_left[:, 3:])

                pos_loss = pos_loss_left + pos_loss_right
                ori_loss = ori_loss_left + ori_loss_right
            else:
                decoder_input = input_batch[:, :14]

                z, recon_mu, z_mu, z_logvar = vae.forward(input_batch, decoder_input)
                kl_loss = vae.kl_loss(z_mu, z_logvar)

                target_batch = to_var(torch.from_numpy(target_batch))
                output = recon_mu

                pos_loss = vae.mse(output[:, :3], target_batch[:, :3])
                ori_loss = vae.rotation_loss(output[:, 3:], target_batch[:, 3:])

            recon_loss = pos_loss + ori_loss

            # recon_loss = vae.recon_loss(output, target_batch)
            # loss = vae.total_loss(output, target_batch, z_mu, z_logvar)
            
            loss = kl_loss + recon_loss
            loss.backward()
            vae.optimizer.step()

            epoch_total_loss = epoch_total_loss + loss.data
            epoch_kl_loss = epoch_kl_loss + kl_loss.data
            epoch_pos_loss = epoch_pos_loss + pos_loss.data
            epoch_ori_loss = epoch_ori_loss + ori_loss.data
            epoch_recon_loss = epoch_recon_loss + recon_loss.data

            running_ori_loss.append(ori_loss_right.data)
            running_pos_loss.append(pos_loss_right.data)
            running_ori_outputs.append(output_r[:, 3:].data)
            running_pos_outputs.append(output_r[:, :3].data)

            if (i/batch_size) % args.batch_freq == 0:
                # print('Train Epoch: %d [%d/%d (%f)]\tLoss: %f\tKL: %f\tPos: %f\t Ori: %f' % (
                #        epoch, i, dataset_size,
                #        100.0 * i / dataset_size/batch_size,
                #        loss.item(),
                #        kl_loss.item(),
                #        pos_loss.item(),
                #        ori_loss.item()))
                print('Train Epoch: %d [%d/%d (%f)]\tLoss: %f\tKL: %f\tR Pos: %f\t R Ori: %f\tL Pos: %f\tL Ori: %f' % (
                       epoch, i, dataset_size,
                       100.0 * i / dataset_size/batch_size,
                       loss.item(),
                       kl_loss.item(),
                       pos_loss_right.item(),
                       ori_loss_right.item(),
                       pos_loss_left.item(),
                       ori_loss_left.item()))
        np.savez(
            os.path.join(
                trained_model_path,
                args.model_name+'_epoch_'+str(epoch) + '_recon_loss.npz'),
            ori_loss=np.asarray(running_ori_loss),
            pos_loss=np.asarray(running_pos_loss),
            ori_outputs=np.asarray(running_ori_outputs),
            pos_outputs=np.asarray(running_pos_outputs))                       
        print(' --avgerage loss: ')
        print(epoch_total_loss/(dataset_size/batch_size))
        loss_dict = {
            'epoch_total': epoch_total_loss/(dataset_size/batch_size),
            'epoch_kl': epoch_kl_loss/(dataset_size/batch_size),
            'epoch_pos': epoch_pos_loss/(dataset_size/batch_size),
            'epoch_ori': epoch_ori_loss/(dataset_size/batch_size),
            'epoch_recon': epoch_recon_loss/(dataset_size/batch_size)
        }
        total_loss.append(loss_dict)

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
                        default=3e-4)
    parser.add_argument('--model_path', type=str,
                        default='/root/training/saved_models')
    parser.add_argument('--start_epoch', type=int,
                        default=0)
    parser.add_argument('--save_freq', type=int,
                        default=1)
    parser.add_argument('--batch_freq', type=int,
                        default=3)
    parser.add_argument('--start_rep', type=str,
                        default='pose')
    parser.add_argument('--model_name', type=str)

    args = parser.parse_args()
    main(args)
