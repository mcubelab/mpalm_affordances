import os, os.path as osp
import sys
import argparse
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

# from airobot.utils import common

from data import SkillPlayDataset, SkeletonDatasetGlamor
from glamor.models import MultiStepDecoder, InverseModel
from skeleton_utils.utils import prepare_sequence_tokens
from skeleton_utils.language import SkillLanguage  
from skeleton_utils.skeleton_globals import PAD_token, SOS_token, EOS_token
from skeleton_utils.glamor_utils import get_uniform_sample_inds, get_boltzmann_sample_inds, filter_skills_probs

import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/config/explore_cfgs'))
from default_skill_names import get_skillset_cfg


def pad_collate(batch):
  x_lens = [len(x) for x in batch]
  xx_pad = pad_sequence(batch, batch_first=True, padding_value=0)

  return xx_pad, x_lens


def eval_ratio(dataloader, model, prior_model, inverse_model, language, args, logdir, sampling='boltzmann'):
    model.eval()
    prior_model.eval()
    inverse_model.eval()

    if args.cuda:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    model = model.to(dev)
    prior_model = prior_model.to(dev)
    inverse_model = inverse_model.to(dev)

    with torch.no_grad():
        for sample in dataloader:
            subgoal, contact, observation, next_observation, action_seq = sample

            bs = subgoal.size(0)

            token_seq = []
            for i, seq in enumerate(action_seq):
                tok = prepare_sequence_tokens(seq.split(' '), language.skill2index)
                token_seq.append(tok)

            observation = observation.float().to(dev)
            next_observation = next_observation.float().to(dev)
            subgoal = subgoal.float().to(dev)
            task_emb = inverse_model(observation, next_observation, subgoal)
            padded_seq_batch = torch.nn.utils.rnn.pad_sequence(token_seq, batch_first=True).to(dev)

            loss = 0

            for j in range(bs):
                target = token_seq[j].to(dev)
                target_skills = [language.index2skill[x.item()] for x in target]
                
                # use inverse model to get overall task encoding
                task_emb = inverse_model(observation[j][None, :, :], next_observation[j][None, :, :], subgoal[j][None, :])
                prior_emb = inverse_model.prior_forward(observation[j][None, :, :])

                # predict skeleton, up to max length
                decoder_input = torch.Tensor([[SOS_token]]).long().to(dev)
                decoder_hidden = task_emb[None, :]
                p_decoder_input = torch.Tensor([[SOS_token]]).long().to(dev)
                p_decoder_hidden = prior_emb[None, :]

                # sample uniformly for skill sequences
                if sampling not in ['boltzmann', 'uniform']:
                    sampling = 'boltzmann'
                if sampling == 'uniform':
                    # skill_inds = get_uniform_sample_inds(len(language.index2skill.keys()), N=args.n_samples, max_steps=args.max_seq_length)
                    skill_inds = get_uniform_sample_inds(len(language.index2skill.keys()), N=args.n_samples, max_steps=padded_seq_batch.size(1))
                    skill_inds = torch.from_numpy(skill_inds).long().to(dev)
                else:
                    skill_inds = get_boltzmann_sample_inds(
                        model, prior_model, task_emb, prior_emb, N=args.n_samples, max_steps=padded_seq_batch.size(1)
                    )
                # skill_inds = torch.cat((skill_inds, padded_seq_batch[j].repeat(10, 1)), axis=0)
                seq_to_score = torch.cat((decoder_input.repeat(skill_inds.size(0), 1), skill_inds), axis=1)
                p_seq_to_score = torch.cat((p_decoder_input.repeat(skill_inds.size(0), 1), skill_inds), axis=1)

                # forward pass to get embeddings for each sequence
                seq_embed = model.embed(seq_to_score)
                p_seq_embed = prior_model.embed(p_seq_to_score)

                decoder_output, _ = model.gru(seq_embed, decoder_hidden.repeat(1, skill_inds.size(0), 1))
                p_decoder_output, _ = prior_model.gru(p_seq_embed, p_decoder_hidden.repeat(1, skill_inds.size(0), 1))

                # get logits from output embeddings
                output_logprobs = model.log_softmax(model.out(decoder_output).permute(0, 2, 1)).permute(0, 2, 1)
                p_output_logprobs = prior_model.log_softmax(prior_model.out(p_decoder_output).permute(0, 2, 1)).permute(0, 2, 1)
                output_logprobs_g = torch.gather(output_logprobs[:, 1:, :], -1, skill_inds[:, :, None]).squeeze()
                p_output_logprobs_g = torch.gather(p_output_logprobs[:, 1:, :], -1, skill_inds[:, :, None]).squeeze()
                
                # filter any that we know are bad
                filtered_skills, output_logprobs_g, p_output_logprobs_g = filter_skills_probs(
                    skill_inds,
                    output_logprobs_g,
                    p_output_logprobs_g,
                    logprob_thresh=args.logprob_thresh
                )

                output_probs, p_output_probs = torch.exp(output_logprobs_g), torch.exp(p_output_logprobs_g)

                # all_valid_skills = []
                # for i in range(filtered_skills.size(0)):
                #     ind_seq = filtered_skills[i]
                #     seq = []
                #     for j in ind_seq:
                #         seq.append(language.index2skill[j.item()])
                #     all_valid_skills.append(seq)

                if output_probs.size(0) > 0:
                    # get likelihood ratios and score
                    # compute product of the likelihoods
                    inverse_prod, prior_prod = output_probs.prod(-1), p_output_probs.prod(-1)

                    # ratio_objective = decoder_output[:, -1] / p_decoder_output[:, -1]
                    ratio_objective = inverse_prod / prior_prod
                    ratio_argmax = ratio_objective.topk(1, dim=0)[-1].item()
                    inverse_argmax = inverse_prod.topk(1, dim=0)[-1].item()
                    
                    best_seq_ratio = filtered_skills[ratio_argmax, :].squeeze()
                    best_seq_inverse = filtered_skills[inverse_argmax, :].squeeze()
                    decoded_skills_ratio = []
                    decoded_skills_inverse = []
                    for t in range(best_seq_ratio.size(0)):
                        decoded_skills_ratio.append(language.index2skill[best_seq_ratio[t].item()])
                    for t in range(best_seq_inverse.size(0)):
                        decoded_skills_inverse.append(language.index2skill[best_seq_inverse[t].item()])
                    print('decoded (ratio): ', decoded_skills_ratio)
                    print('decoded (inverse): ', decoded_skills_inverse)
                    print('target: ', target_skills)
                    print('\n')

def eval(dataloader, model, inverse_model, language, args, logdir, prior=False):
    model.eval()
    inverse_model.eval()

    if args.cuda:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    model = model.to(dev)
    inverse_model = inverse_model.to(dev)

    with torch.no_grad():
        for sample in dataloader:
            subgoal, contact, observation, next_observation, action_seq = sample

            bs = subgoal.size(0)

            token_seq = []
            for i, seq in enumerate(action_seq):
                tok = prepare_sequence_tokens(seq.split(' '), language.skill2index)
                token_seq.append(tok)

            observation = observation.float().to(dev)
            next_observation = next_observation.float().to(dev)
            subgoal = subgoal.float().to(dev)
            if prior:
                task_emb = inverse_model.prior_forward(observation)
            else:
                task_emb = inverse_model(observation, next_observation, subgoal)
            padded_seq_batch = torch.nn.utils.rnn.pad_sequence(token_seq, batch_first=True).to(dev)

            loss = 0

            for j in range(bs):
                decoder_input = torch.Tensor([[SOS_token]]).long().to(dev)
                decoder_hidden = task_emb[j][None, None, :]
                target = token_seq[j].to(dev)
                target_skills = [language.index2skill[x.item()] for x in target]

                decoded_skills = []
                for t in range(target.size(0)):
                    decoder_input = model.embed(decoder_input)
                    decoder_output, decoder_hidden = model.gru(decoder_input, decoder_hidden)
                    output = model.log_softmax(model.out(decoder_output[:, 0]))
                    loss += model.criterion(output, target[t][None])
                    topv, topi = output.topk(1, dim=1)
                    decoder_input = topi            
                    
                    if topi.item() == language.skill2index['EOS']:
                        decoded_skills.append('EOS')
                        break
                    else:
                        decoded_skills.append(language.index2skill[topi.item()])
                print('decoded: ', decoded_skills)
                print('target: ', target_skills)
                print('\n')


def train(model, prior_model, inverse_model, dataloader, test_dataloader, optimizer, language, args, logdir, eval_output=False, iterations=None):
    model.train()
    prior_model.train()
    inverse_model.train()

    if args.cuda:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    model = model.to(dev)
    prior_model = prior_model.to(dev)
    inverse_model = inverse_model.to(dev)

    if iterations is None:
        iterations = 0
    else:
        iterations = iterations
    for epoch in range(args.num_epoch):
        for sample in dataloader:
            iterations += 1
            subgoal, contact, observation, next_observation, action_seq, scene_context = sample
            # observations, action_str, tables, mask, reward_step, transformation, goal = sample 

            bs = subgoal.size(0)
            token_seq = []
            for i, seq in enumerate(action_seq):
                tok = prepare_sequence_tokens(seq.split(' '), language.skill2index)
                token_seq.append(tok)

            observation = observation.float().to(dev)
            next_observation = next_observation.float().to(dev)
            subgoal = subgoal.float().to(dev)
            scene_context = scene_context.float().to(dev)
            # task_emb = inverse_model(observation, next_observation, subgoal)
            # prior_emb = inverse_model.prior_forward(observation)
            task_emb = inverse_model(observation, next_observation, subgoal, scene_context)
            prior_emb = inverse_model.prior_forward(observation, scene_context)
            padded_seq_batch = torch.nn.utils.rnn.pad_sequence(token_seq, batch_first=True, padding_value=EOS_token).to(dev)
            
            decoder_input = torch.Tensor([[SOS_token]]).repeat((padded_seq_batch.size(0), 1)).long().to(dev)
            decoder_hidden = task_emb[None, :, :]
            p_decoder_input = torch.Tensor([[SOS_token]]).repeat((padded_seq_batch.size(0), 1)).long().to(dev)
            p_decoder_hidden = prior_emb[None, :, :]
            inverse_loss = 0
            prior_loss = 0
            max_seq_length = padded_seq_batch.size(1)
            for t in range(max_seq_length):
                decoder_input = model.embed(decoder_input)
                decoder_output, decoder_hidden = model.gru(decoder_input, decoder_hidden)
                output = model.log_softmax(model.out(decoder_output[:, 0])).view(bs, -1)
                topv, topi = output.topk(1, dim=1)
                decoder_input = topi

                p_decoder_input = prior_model.embed(p_decoder_input)
                p_decoder_output, p_decoder_hidden = prior_model.gru(p_decoder_input, p_decoder_hidden)
                p_output = prior_model.log_softmax(prior_model.out(p_decoder_output[:, 0])).view(bs, -1)
                p_topv, p_topi = p_output.topk(1, dim=1)
                p_decoder_input = p_topi            

                inverse_loss += model.criterion(output, padded_seq_batch[:, t])
                prior_loss += prior_model.criterion(p_output, padded_seq_batch[:, t])
            loss = inverse_loss + prior_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iterations % args.log_interval == 0:
                kvs = {}
                kvs['loss'] = loss.item()
                kvs['inverse_loss'] = inverse_loss.item()
                kvs['prior_loss'] = prior_loss.item()
                log_str = 'Epoch: {}, Iteration: {}, '.format(epoch, iterations)
                for k, v in kvs.items():
                    log_str += '%s: %.5f, ' % (k,v)
                print(log_str)

            if iterations % args.save_interval == 0:
                model = model.eval()
                prior_model = prior_model.eval()
                inverse_model = inverse_model.eval()
                model_path = osp.join(logdir, "model_{}".format(iterations))
                torch.save({'model_state_dict': model.state_dict(),
                            'prior_model_state_dict': prior_model.state_dict(),
                            'inverse_model_state_dict': inverse_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'language_idx2skill': language.index2skill,
                            'language_skill2idx': language.skill2index,
                            'args': args}, model_path)
                print("Saving model in directory....")


                ## running test
                if eval_output:
                    # print('run test')
                    # print('running test on test examples, with no likelihood ratio scoring')
                    # eval(test_dataloader, model, inverse_model, language, args, logdir, prior=False)
                    # eval(test_dataloader, prior_model, inverse_model, language, args, logdir, prior=True)
                    # eval_ratio(test_dataloader, model, prior_model, inverse_model, language, args, logdir)

                    # print('running test on training examples with likelihood ratio scoring')
                    # eval_ratio(dataloader, model, prior_model, inverse_model, language, args, logdir)

                    print('running test on test examples with likelihood ratio scoring')
                    eval_ratio(test_dataloader, model, prior_model, inverse_model, language, args, logdir)
                model = model.train()   
                prior_model = prior_model.train()
                inverse_model = inverse_model.train()
    return iterations


def main(args):
    # train_data = SkillPlayDataset('train', aug=True, max_steps=4)
    # test_data = SkillPlayDataset('test', aug=True, max_steps=4) 

    # train_data = SkeletonDatasetGlamor('train')
    # test_data = SkeletonDatasetGlamor('test') 

    train_data = SkeletonDatasetGlamor('train', append_table=True)
    test_data = SkeletonDatasetGlamor('test', append_table=True) 
    # train_data = SkeletonDataset('overfit')
    # test_data = SkeletonDataset('overfit')      

    skill_lang = SkillLanguage('default')

    skillset_cfg = get_skillset_cfg()
    for skill_name in skillset_cfg.SKILL_SET:
        skill_lang.add_skill(skill_name)

    # language_loader = DataLoader(train_data, batch_size=1)
    # for sample in language_loader:
    #     # seq = sample[1]
    #     seq = sample[-1]
    #     skill_lang.add_skill_seq(seq[0])
    print('Skill Language: ')
    print(skill_lang.skill2index, skill_lang.index2skill)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    
    hidden_dim = args.latent_dim

    # in_dim is [x, y, z, x0, y0, z0]
    # out_dim mukst include total number of skills, pad token, and eos token
    in_dim = 6
    # out_dim = 9
    out_dim = len(skill_lang.index2skill.keys())

    inverse = InverseModel(in_dim, in_dim, hidden_dim, hidden_dim)
    model = MultiStepDecoder(hidden_dim, out_dim)
    prior_model = MultiStepDecoder(hidden_dim, out_dim)
    params = list(inverse.parameters()) + list(model.parameters()) + list(prior_model.parameters())

    optimizer = optim.Adam(params, lr=args.lr)

    logdir = osp.join(args.logdir, args.exp)
    if not osp.exists(logdir):
        os.makedirs(logdir)

    if args.resume_iter != 0:
        model_path = osp.join(logdir, "model_{}".format(args.resume_iter))
        checkpoint = torch.load(model_path)
        args_old = checkpoint['args']

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        prior_model.load_state_dict(checkpoint['prior_model_state_dict'])
        inverse.load_state_dict(checkpoint['inverse_model_state_dict'])
    
    if args.train:
        train(model, prior_model, inverse, train_loader, test_loader, optimizer, skill_lang, args, logdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='whether or not to train')
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not')

    # Generic Parameters for Experiments
    parser.add_argument('--dataset', default='intphys', type=str, help='intphys or others')
    parser.add_argument('--logdir', default='glamor_cachedir', type=str, help='location where log of experiments will be stored')
    parser.add_argument('--rundir', default='runs/glamor_runs')
    parser.add_argument('--exp', default='glamor_debug', type=str, help='name of experiments')
    parser.add_argument('--data_workers', default=4, type=int, help='Number of different data workers to load data in parallel')

    # train
    parser.add_argument('--batch_size', default=128, type=int, help='size of batch of input to use')
    parser.add_argument('--num_epoch', default=10000, type=int, help='number of epochs of training to run')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for training')
    parser.add_argument('--log_interval', default=10, type=int, help='log outputs every so many batches')
    parser.add_argument('--save_interval', default=1000, type=int, help='save outputs every so many batches')
    parser.add_argument('--resume_iter', default=0, type=str, help='iteration to resume training')

    # model 
    parser.add_argument('--latent_dim', default=512, type=int, help='size of hidden representation')
    parser.add_argument('--max_seq_length', default=5, type=int, help='maximum sequence length')
    parser.add_argument('--n_samples', default=50, type=int, help='maximum sequence length')
    parser.add_argument('--logprob_thresh', default=-10.0, type=float)
    parser.add_argument('--prob_thresh', default=1e-5, type=float)

    args = parser.parse_args()

    main(args)
