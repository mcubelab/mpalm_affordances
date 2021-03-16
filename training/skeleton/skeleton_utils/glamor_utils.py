import sys
import numpy as np
import torch

sys.path.append('..')
from skeleton_utils.skeleton_globals import SOS_token, EOS_token, PAD_token

def get_uniform_sample_inds(K, N, max_steps):
    """
    Function to create a numpy array containing the categorical indices
    of each skill, after uniformly sampling for them and cutting each
    sampled sequence off once EOS has been sampled.

    Args:
        K (int): Total number of skills
        N (int): Total number of sequences to sample
        max_steps (int): Maximum number of steps to sample
    """
    skill_inds = np.random.randint(low=EOS_token, high=K, size=(N, max_steps))
    skill_inds[np.where(skill_inds == SOS_token)] 
    idx0, idx1 = np.where(skill_inds == EOS_token)
    last0 = -1
    for i in range(idx0.shape[0]):
        if idx0[i] == last0:
            continue
        skill_inds[idx0[i], idx1[i]+1:] = PAD_token
        last0 = idx0[i]
    return skill_inds


def get_boltzmann_sample_inds(model, prior_model, task_embed, prior_embed, N, max_steps):
    """
    Function to auto-regressively sample candidate skill sequences from
    a Boltzmann distribution parameterized by the ratio between the inverse
    model and prior model at each step

    Args:
        model (torch.nn.Module): Inverse model
        prior_model (torch.nn.Module): Prior model
        task_embed (torch.Tensor): Embedding containing start/goal information
        prior_embed (torch.Tensor): Embedding containing only goal information
        N (int): Total number of sequences to sample
        max_steps (int): Maximum number of steps to sample
    """
    # from IPython import embed
    # embed()
    dev = next(model.parameters()).device

    # predict skeleton, up to max length
    decoder_input_start = torch.Tensor([[SOS_token]]).long().to(dev)
    decoder_hidden = task_embed[None, :]
    p_decoder_input_start = torch.Tensor([[SOS_token]]).long().to(dev)
    p_decoder_hidden = prior_embed[None, :]

    decoder_input = decoder_input_start.repeat((N, 1)).long().to(dev)
    decoder_hidden = decoder_hidden.repeat((1, N, 1))
    p_decoder_input = p_decoder_input_start.repeat((N, 1)).long().to(dev)
    p_decoder_hidden = p_decoder_hidden.repeat((1, N, 1))
    start_s = decoder_input.size()

    candidate_skills = torch.empty((N, max_steps)).long().to(dev) 
    for t in range(max_steps):
        # get predictions from model that takes both start and goal
        decoder_input = model.embed(decoder_input)
        decoder_output, decoder_hidden = model.gru(decoder_input, decoder_hidden)
        output = model.log_softmax(model.out(decoder_output[:, 0]))

        # get predictions from model that only takes start (p_ indicated prior)
        p_decoder_input = prior_model.embed(p_decoder_input)
        p_decoder_output, p_decoder_hidden = prior_model.gru(p_decoder_input, p_decoder_hidden)
        p_output = prior_model.log_softmax(prior_model.out(p_decoder_output[:, 0]))
        
        # get z scores and construct distribution to sample from
        output_probs, p_output_probs = torch.exp(output), torch.exp(p_output)
        z = output_probs / p_output_probs
        Q = torch.sum(torch.exp(-1.0 / z), axis=1)
        # probs = torch.exp(z) / Q[:, None].repeat((1, z.size(1)))
        probs = torch.exp(-1.0 / z) / Q[:, None].repeat((1, z.size(1)))
        m = torch.distributions.Categorical(probs=probs)
        next_skill = m.sample(decoder_input_start.size())
        candidate_skills[:, t] = next_skill.squeeze()
        decoder_input, p_decoder_input = next_skill.view(start_s).long(), next_skill.view(start_s).long()

    idx0, idx1 = torch.where(candidate_skills == EOS_token)
    last0 = -1
    for i in range(idx0.shape[0]):
        if idx0[i] == last0:
            continue
        candidate_skills[idx0[i], idx1[i]+1:] = EOS_token
        last0 = idx0[i]
    return candidate_skills


def filter_skills_probs(skill_inds, output_logprobs_g, p_output_logprobs_g, logprob_thresh):
    # above probability threshold
    above_thresh_inds = torch.where(torch.prod(output_logprobs_g > logprob_thresh, 1))
    output_logprobs_g, p_output_logprobs_g = output_logprobs_g[above_thresh_inds], p_output_logprobs_g[above_thresh_inds]
    filtered_skills = skill_inds[above_thresh_inds]

    # doesn't start with EOS
    no_eos_start_inds = torch.where(filtered_skills[:, 0] != EOS_token)
    output_logprobs_g, p_output_logprobs_g = output_logprobs_g[no_eos_start_inds], p_output_logprobs_g[no_eos_start_inds]
    filtered_skills = filtered_skills[no_eos_start_inds]

    # doesn't contain PAD or SOS
    no_sos_pad = torch.where((torch.prod(filtered_skills != PAD_token, 1)) & (torch.prod(filtered_skills != SOS_token, 1)))
    output_logprobs_g, p_output_logprobs_g = output_logprobs_g[no_sos_pad], p_output_logprobs_g[no_sos_pad]
    filtered_skills = filtered_skills[no_sos_pad]
    return filtered_skills, output_logprobs_g, p_output_logprobs_g