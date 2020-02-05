import os
import argparse
import time
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from data_loader import DataLoader
from model import VAE, GoalVAE
from util import to_var, save_state, load_net_state, load_seed, load_opt_state, load_args


def main(args):
    batch_size = args.batch_size
    dataset_size = args.total_data_size
    if not hasattr(args, 'experiment_name'):
        experiment_name = args.model_name
    else:
        experiment_name = args.experiment_name

    if not os.path.exists(os.path.join(args.log_dir, experiment_name)):
        os.makedirs(os.path.join(args.log_dir, experiment_name))

    writer = SummaryWriter(os.path.join(args.log_dir, experiment_name))

    # torch_seed = np.random.randint(low=0, high=1000)
    # np_seed = np.random.randint(low=0, high=1000)
    torch_seed = 0
    np_seed = 0

    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)

    trained_model_path = os.path.join(args.model_path, args.model_name)
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)

    if args.task == 'contact':
        if args.skill_type == 'pull':
            decoder_input_dim = args.input_dimension - args.output_dimension
        elif args.skill_type == 'grasp':
            decoder_input_dim = args.input_dimension - 2*args.output_dimension
        vae = VAE(
            args.input_dimension,
            args.output_dimension,
            args.latent_dimension,
            decoder_input_dim,
            lr=args.learning_rate
        )
    elif args.task == 'goal':
        if args.start_rep == 'keypoints':
            start_dim = 24
        elif args.start_rep == 'pose':
            start_dim = 7

        if args.goal_rep == 'keypoints':
            goal_dim = 24
        elif args.goal_rep == 'pose':
            goal_dim = 7

        input_dim = start_dim + goal_dim
        output_dim = goal_dim
        decoder_input_dim = start_dim
        vae = GoalVAE(
            input_dim,
            output_dim,
            args.latent_dimension,
            decoder_input_dim,
            lr=args.learning_rate
        )
    elif args.task == 'transformation':
        input_dim = 7
        output_dim = 7
        decoder_input_dim = 0
        vae = GoalVAE(
            input_dim,
            output_dim,
            args.latent_dimension,
            decoder_input_dim,
            lr=args.learning_rate
        )
    else:
        raise ValueError('training task not recognized')

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
        args = load_args(fname)
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

            input_batch, decoder_input_batch, target_batch = \
                data_loader.load_batch(i, batch_size, start_rep=args.start_rep, goal_rep=args.goal_rep, task=args.task)
            input_batch = to_var(torch.from_numpy(input_batch))
            decoder_input_batch = to_var(torch.from_numpy(decoder_input_batch))

            z, recon_mu, z_mu, z_logvar = vae.forward(input_batch, decoder_input_batch)
            kl_loss = vae.kl_loss(z_mu, z_logvar)
            # output_r, output_l = recon_mu

            if args.task == 'contact':
                output_r, output_l = recon_mu
                if args.skill_type == 'grasp':
                    target_batch_right = to_var(torch.from_numpy(target_batch[:, 0]))
                    target_batch_left = to_var(torch.from_numpy(target_batch[:, 1]))

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
                elif args.skill_type == 'pull':
                    target_batch = to_var(torch.from_numpy(target_batch.squeeze()))

                    #TODO add flags for when we're training both arms
                    # output = recon_mu[0]  # right arm is index [0]
                    # output = recon_mu[1]  # left arm is index [1]

                    pos_loss_right = vae.mse(output_r[:, :3], target_batch[:, :3])
                    ori_loss_right = vae.rotation_loss(output_r[:, 3:], target_batch[:, 3:])

                    pos_loss = pos_loss_right
                    ori_loss = ori_loss_right
                running_ori_loss.append(ori_loss_right.data)
                running_pos_loss.append(pos_loss_right.data)
                running_ori_outputs.append(output_r[:, 3:].data)
                running_pos_outputs.append(output_r[:, :3].data)

            elif args.task == 'goal':
                target_batch = to_var(torch.from_numpy(target_batch.squeeze()))

                output = recon_mu
                if args.goal_rep == 'pose':
                    pos_loss = vae.mse(output[:, :3], target_batch[:, :3])
                    ori_loss = vae.rotation_loss(output[:, 3:], target_batch[:, 3:])
                elif args.goal_rep == 'keypoints':
                    pos_loss = vae.mse(output, target_batch)
                    ori_loss = torch.zeros(pos_loss.shape)

                running_ori_loss.append(ori_loss.data)
                running_pos_loss.append(pos_loss.data)
                running_ori_outputs.append(output[:, 3:].data)
                running_pos_outputs.append(output[:, :3].data)
            elif args.task == 'transformation':
                target_batch = to_var(torch.from_numpy(target_batch.squeeze()))

                output = recon_mu
                pos_loss = vae.mse(output[:, :3], target_batch[:, :3])
                ori_loss = vae.rotation_loss(output[:, 3:], target_batch[:, 3:])

                running_ori_loss.append(ori_loss.data)
                running_pos_loss.append(pos_loss.data)
                running_ori_outputs.append(output[:, 3:].data)
                running_pos_outputs.append(output[:, :3].data)

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

            writer.add_scalar('loss/train/ori_loss', ori_loss.data, i)
            writer.add_scalar('loss/train/pos_loss', pos_loss.data, i)

            if (i/batch_size) % args.batch_freq == 0:
                if args.skill_type == 'pull' or args.task == 'goal':
                    print('Train Epoch: %d [%d/%d (%f)]\tLoss: %f\tKL: %f\tPos: %f\t Ori: %f' % (
                        epoch, i, dataset_size,
                        100.0 * i / dataset_size/batch_size,
                        loss.item(),
                        kl_loss.item(),
                        pos_loss.item(),
                        ori_loss.item()))
                elif args.skill_type == 'grasp' and args.task == 'contact':
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
                args=args,
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
        args=args,
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
    parser.add_argument('--goal_rep', type=str,
                        default='pose')
    parser.add_argument('--log_dir', type=str,
                        default='/root/training/runs')
    parser.add_argument('--skill_type', type=str,
                        default='pull')
    parser.add_argument('--task', type=str,
                        default='contact')

    parser.add_argument('--model_name', type=str)

    args = parser.parse_args()
    main(args)
