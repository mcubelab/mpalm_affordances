import torch
from torch.autograd import Variable


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def save_state(net, torch_seed, np_seed, args, fname):
    states = {
        'encoder_state_dict': net.encoder.state_dict(),
        'decoder_state_dict': net.decoder.state_dict(),
        'optimizer': net.optimizer.state_dict(),
        'torch_seed': torch_seed,
        'np_seed': np_seed,
        'args': args
    }
    torch.save(states, fname)


def load_net_state(net, fname):
    checkpoint = torch.load(
        fname,
        map_location='cuda:%d' % (torch.cuda.current_device()))
    net.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    net.decoder.load_state_dict(checkpoint['decoder_state_dict'])


def load_opt_state(net, fname):
    checkpoint = torch.load(
        fname,
        map_location='cuda:%d' % (torch.cuda.current_device()))
    net.optimizer.load_state_dict(checkpoint['optimizer'])


def load_seed(fname):
    checkpoint = torch.load(
        fname,
        map_location='cuda:%d' % (torch.cuda.current_device()))
    return checkpoint['torch_seed'], checkpoint['np_seed']


def load_args(fname):
    checkpoint = torch.load(
        fname,
        map_location='cuda:%d' % (torch.cuda.current_device()))
    return checkpoint['args']