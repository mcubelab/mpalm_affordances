import os, os.path as osp
import sys
import argparse
import random
import copy
import signal
import time
from torch.multiprocessing import Process, Manager, Pipe, Queue
import torch.multiprocessing as mp
import lcm
from collections import OrderedDict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

# from airobot.utils import common
from airobot import set_log_level, log_debug, log_info, log_warn, log_critical

from data import SkillPlayDataset, SkeletonDatasetGlamor
from glamor.models import MultiStepDecoder, InverseModel
from lcm_inference.skeleton_predictor_lcm import GlamorSkeletonPredictorLCM
from skeleton_utils.utils import prepare_sequence_tokens, process_pointcloud_batch, state_dict_to_cpu, cn2dict
from skeleton_utils.language import SkillLanguage 
from skeleton_utils.skeleton_globals import PAD_token, SOS_token, EOS_token
from skeleton_utils.replay_buffer import TransitionBuffer
from skeleton_utils.buffer_lcm import BufferLCM
from skeleton_utils.server import PredictionServerInit, SkeletonServerParams, serve_wrapper 
from skeleton_utils.results import RPOMetricWriter
from train_glamor import train as pretrain

import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/config/explore_cfgs'))
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/config/experiment_cfgs'))
from default_skill_names import get_skillset_cfg
from rpo_lcm import rpo_plan_skeleton_t
from glamor_explore_defaults import get_glamor_explore_defaults


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


def model_worker(child_conn, work_queue, result_queue, global_dict, seed, worker_id):
    while True:
        try:
            if not child_conn.poll(0.0001):
                continue
            msg = child_conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if msg == "INIT":
            global_dict['prediction_server_ready'] = False
            global_dict['stop_prediction_server'] = False
            log_info('Model worker %d: initializing prediction server' % worker_id)

            # create neural network model
            in_dim = 6
            skill_lang = global_dict['skill_lang']
            hidden_dim = global_dict['hidden_dim']
            model_state_dict = global_dict['model_state_dict']
            inverse_state_dict = global_dict['inverse_state_dict']
            prior_model_state_dict = global_dict['prior_model_state_dict']
            args = global_dict['args']
            if args.debug:
                set_log_level('debug')
            else:
                set_log_level('info')

            out_dim = len(skill_lang.index2skill.keys())

            inverse = InverseModel(in_dim, in_dim, hidden_dim, hidden_dim).cuda()
            model = MultiStepDecoder(hidden_dim, out_dim).cuda()
            prior_model = MultiStepDecoder(hidden_dim, out_dim).cuda()

            model.load_state_dict(model_state_dict)
            prior_model.load_state_dict(prior_model_state_dict)
            inverse.load_state_dict(inverse_state_dict)
            lc = lcm.LCM()

            # create interface to LCM predictor
            lcm_predictor = GlamorSkeletonPredictorLCM(
                lc=lc,
                model=model,
                prior_model=prior_model,
                inverse_model=inverse,
                args=args,
                language=skill_lang,
                verbose=args.verbose)

            global_dict['prediction_server_ready'] = True
            continue
        if msg == "PREDICT":
            log_info('Model worker %d: running prediction server' % worker_id)
            # be running LCM predictor, making predictions when observations are received
            while True: 
                lcm_predictor.predict_skeleton()
                if global_dict['stop_prediction_server']:
                    log_debug('Model worker %d: stopping prediction server' % worker_id)
                    break
            continue
        if msg == "UPDATE":
            log_info('Model worker %d: updating model weights' % worker_id)
            # get updated model weights
            global_dict['prediction_server_ready'] = False
            model_state_dict = global_dict['model_state_dict']
            prior_model_state_dict = global_dict['prior_model_state_dict']
            inverse_state_dict = global_dict['inverse_state_dict']

            model.load_state_dict(model_state_dict)
            prior_model.load_state_dict(prior_model_state_dict)
            inverse.load_state_dict(inverse_state_dict)
            global_dict['prediction_server_ready'] = True
            global_dict['stop_prediction_server'] = False
            continue
        if msg == "END":
            break
        time.sleep(0.001)
    log_info('Breaking worker ID: %d' % worker_id)
    child_conn.close()


class ModelWorkerManager:
    def __init__(self, work_queue, result_queue, global_manager, num_workers=1):
        # thread/process for sending commands to the robot
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.global_manager = global_manager
        self.global_dict = self.global_manager.dict()
        self.worker_flag_dict = self.global_manager.dict()        

        self.np_seed_base = 1
        self.setup_workers(num_workers)

    def setup_workers(self, num_workers):
        """Setup function to instantiate the desired number of
        workers. Pipes and Processes set up, stored internally,
        and started.
        Args:
            num_workers (int): Desired number of worker processes
        """
        worker_ids = np.arange(num_workers, dtype=np.int64).tolist()
        seeds = np.arange(self.np_seed_base, self.np_seed_base + num_workers, dtype=np.int64).tolist()

        self._worker_ids = worker_ids
        self.seeds = seeds

        self._pipes = {}
        self._processes = {}
        for i, worker_id in enumerate(self._worker_ids):
            parent, child = Pipe(duplex=True)
            self.worker_flag_dict[worker_id] = True
            proc = Process(
                target=model_worker,
                args=(
                    child,
                    self.work_queue,
                    self.result_queue,
                    self.global_dict,
                    seeds[i],
                    worker_id,
                )
            )
            pipe = {}
            pipe['parent'] = parent
            pipe['child'] = child

            self._pipes[worker_id] = pipe
            self._processes[worker_id] = proc

    def init_workers(self):
        for i, worker_id in enumerate(self._worker_ids):
            self._processes[worker_id].start()
            self._pipes[worker_id]['parent'].send('INIT')
            print('RESET WORKER ID: ' + str(worker_id))
        print('FINISHED WORKER SETUP')

    def run_predictions(self):
        while True:
            if self.global_dict['prediction_server_ready']:
                break
            time.sleep(0.001)
        for i, worker_id in enumerate(self._worker_ids):
            self._pipes[worker_id]['parent'].send('PREDICT')

    def update_weights(self):
        """
        Function to send a signal to the workers indicating to stop making predictions
        and update the internal weights that are being used. Procedure for this is as follows
        1. global_dict with shared memory has 'model_state_dict' and 'inverse_state_dict' fields
            updated with the weights to load to the prediction models
        2. call this function from the main process to stop the predictions and update the weights
        3. call run_predictions() to begin the prediction process again, now with the new weights
        """
        for i, worker_id in enumerate(self._worker_ids):
            self.global_dict['stop_prediction_server'] = True
            self._pipes[worker_id]['parent'].send('UPDATE')

def train(model, prior_model, inverse_model, buffer, optimizer, language, args, logdir, iterations=None):
    model.train()
    prior_model.train()
    inverse_model.train()

    if args.cuda:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    model = model.to(dev)
    prior_model = model.to(dev)
    inverse_model = inverse_model.to(dev)

    # iterations = 0
    # global iterations
    if iterations is None:
        iterations = 0
    else:
        iterations = iterations
    log_info('Training for %d epochs with %d samples in the buffer' % (args.num_epoch, buffer.index))
    for epoch in range(args.num_epoch):
        sample = buffer.sample_sg(n=args.batch_size)
        # for sample in buffer_samples:
        iterations += 1

        subgoal, contact, observation, next_observation, action_seq, scene_context = sample

        bs = subgoal.size(0)
        
        observation = observation.float().to(dev)
        next_observation = next_observation.float().to(dev)
        scene_context = scene_context.float().to(dev)
        observation = process_pointcloud_batch(observation).to(dev)
        next_observation = process_pointcloud_batch(next_observation).to(dev)
        scene_context = process_pointcloud_batch(scene_context).to(dev)
        subgoal = subgoal.float().to(dev)
        # task_emb = inverse_model(observation, next_observation, subgoal)
        # prior_emb = inverse_model.prior_forward(observation)
        task_emb = inverse_model(observation, next_observation, subgoal, scene_context)
        prior_emb = inverse_model.prior_forward(observation, scene_context)
        
        # padded_seq_batch = torch.nn.utils.rnn.pad_sequence(token_seq, batch_first=True).to(dev)
        padded_seq_batch = action_seq.squeeze()
        
        decoder_input = torch.Tensor([[SOS_token]]).repeat((padded_seq_batch.size(0), 1)).long().to(dev)
        decoder_hidden = task_emb[None, :, :]
        p_decoder_input = torch.Tensor([[SOS_token]]).long().repeat((padded_seq_batch.size(0), 1)).long().to(dev)
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
            # loss += model.criterion(output, padded_seq_batch[:, t].squeeze())

            # get predictions from model that only takes start (p_ indicated prior)
            p_decoder_input = prior_model.embed(p_decoder_input)
            p_decoder_output, p_decoder_hidden = prior_model.gru(p_decoder_input, p_decoder_hidden)
            p_output = prior_model.log_softmax(prior_model.out(p_decoder_output[:, 0])).view(bs, -1)
            p_topv, p_topi = p_output.topk(1, dim=1)
            p_decoder_input = p_topi            

            inverse_loss += model.criterion(output, padded_seq_batch[:, t])
            prior_loss += prior_model.criterion(p_output, padded_seq_batch[:, t])
        loss = inverse_loss + args.alpha*prior_loss
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
            model_path = osp.join(logdir, "model_{}".format(iterations))
            torch.save({'model_state_dict': model.state_dict(),
                        'prior_model_state_dict': prior_model.state_dict(),
                        'inverse_model_state_dict': inverse_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'language_idx2skill': language.index2skill,
                        'language_skill2idx': language.skill2index,
                        'args': args}, model_path)
            print("Saving model in directory....")
    return iterations



def main(args):
    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')
    mp.set_start_method('spawn')
    signal.signal(signal.SIGINT, signal_handler)
    lc = lcm.LCM()
    experiment_cfg = get_glamor_explore_defaults()
    exp_run_cfg_file = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/config/experiment_cfgs/run_cfgs', args.exp + '.yaml')
    if not osp.exists(exp_run_cfg_file):
        import yaml
        with open(exp_run_cfg_file, 'w') as f:
            yaml.dump(cn2dict(experiment_cfg), f)
        input('\n\nWaiting for setting config for this run at file: %s, press enter to proceed\n\n' % exp_run_cfg_file)
    experiment_cfg.merge_from_file(exp_run_cfg_file)
    experiment_cfg.freeze()

    # overwrite args with specific experiment configs
    args.pretrain = experiment_cfg.pretraining.use_pretraining
    args.num_pretrain_epoch = experiment_cfg.pretraining.num_pretrain_epoch

    buffer = TransitionBuffer(
        size=5000,
        observation_n=(100, 3),
        context_n=(100, 3),
        action_n=1,
        device=torch.device('cuda:0'),
        goal_n=7)

    buffer_to_lcm = BufferLCM(lc, buffer, new_msgs_max=args.env_episodes_to_add)

    train_data = SkeletonDatasetGlamor('train', append_table=True)
    test_data = SkeletonDatasetGlamor('test', append_table=True) 

    skill_lang = SkillLanguage('default')

    skillset_cfg = get_skillset_cfg()
    for skill_name in skillset_cfg.SKILL_SET:
        skill_lang.add_skill(skill_name)
    print('Skill Language: ')
    print(skill_lang.skill2index, skill_lang.index2skill)
    server_params = SkeletonServerParams()
    server_params.set_skill2index(skill_lang.skill2index)
    server_params.set_experiment_name(args.exp)
    server_params.set_experiment_config(cn2dict(experiment_cfg))
    server_params.set_train_args(args)

    server_proc = Process(target=serve_wrapper, args=(server_params,))
    server_proc.daemon = True
    server_proc.start()
    log_info('Starting Skeleton Prediction Parameter Server')

    rundir = osp.join(args.rundir, args.exp)
    if not osp.exists(rundir):
        os.makedirs(rundir)
    metric_writer = RPOMetricWriter(exp_name=args.exp, rundir=rundir)
    metric_writer.collector_process.start()

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    
    hidden_dim = args.latent_dim

    # in_dim is [x, y, z, x0, y0, z0]
    # out_dim mukst include total number of skills, pad token, and eos token
    in_dim = 6
    in_context_dim = 6
    # out_dim = 9
    # out_dim = len(skill_lang.skill2index.keys())
    out_dim = len(skill_lang.index2skill.keys())

    inverse = InverseModel(in_dim, in_context_dim, hidden_dim, hidden_dim).cuda()
    model = MultiStepDecoder(hidden_dim, out_dim).cuda()
    prior_model = MultiStepDecoder(hidden_dim, out_dim).cuda()
    params = list(inverse.parameters()) + list(model.parameters()) + list(prior_model.parameters())

    optimizer = optim.Adam(params, lr=args.lr)

    logdir = osp.join(args.logdir, args.exp)
    if not osp.exists(logdir):
        os.makedirs(logdir)

    # TODO: handle the model_path variable in non-pretrainin mode better
    # model_path = osp.join(logdir, "model_{}".format(args.resume_iter))
    # global iterations
    iterations = 0
    if args.resume_iter != 0:
        model_path = osp.join(logdir, "model_{}".format(args.resume_iter))
        checkpoint = torch.load(model_path)
        args_old = checkpoint['args']
        for kwarg, value in args.__dict__.items():
            if kwarg not in args_old.__dict__.keys():
                args_old.__dict__[kwarg] = value 
            if kwarg not in ['pretrain', 'batch_size', 'resume_iter']:
                args.__dict__[kwarg] = args_old.__dict__[kwarg]

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        inverse.load_state_dict(checkpoint['inverse_model_state_dict'])
        log_info('Setting iterations value to %d' % int(args.resume_iter))
        iterations = int(args.resume_iter)

        # skill_lang.index2skill = checkpoint['language_idx2skill']
        # skill_lang.skill2index = checkpoint['language_skill2idx']
            
    if args.pretrain:
        log_info('Glamor exploration: pretraining on static data')
        args.num_epoch = copy.deepcopy(args.num_pretrain_epoch)
        iterations = pretrain(model, prior_model, inverse, train_loader, test_loader, optimizer, skill_lang, args, logdir, iterations=iterations)

    print('Starting training at iteration: %d' % iterations)
    # set up processes
    work_queue = Queue()
    result_queue = Queue()
    mp_manager = Manager()
    manager = ModelWorkerManager(work_queue, result_queue, mp_manager, num_workers=1)

    manager.global_dict['prediction_server_ready'] = False
    manager.global_dict['stop_prediction_server'] = False
    manager.global_dict['skill_lang'] = skill_lang
    manager.global_dict['hidden_dim'] = hidden_dim

    # multiprocessing doesn't like sharing CUDA tensors
    manager.global_dict['model_state_dict'] = state_dict_to_cpu(model.state_dict()) 
    manager.global_dict['prior_model_state_dict'] = state_dict_to_cpu(prior_model.state_dict()) 
    manager.global_dict['inverse_state_dict'] = state_dict_to_cpu(inverse.state_dict())
    # manager.global_dict['model_path'] = model_path
    manager.global_dict['args'] = args
    
    log_info('Glamor exploration: initializing prediction server')
    manager.init_workers()
    manager.run_predictions()
    buffer_to_lcm.start_buffer_thread()

    log_info('Glamor exploration: beginning LCM loop')
    log_debug('Glamor exploration: receiving data and adding to buffer')
    while True:
        try:
            # send some data over 
            time.sleep(0.001)
            # if not buffer_to_lcm.new_msgs >= args.env_episodes_to_add:
            if buffer_to_lcm.max_overflow_count == 0:
                continue
            buffer_to_lcm.max_overflow_count -= 1
            
            # train the models
            log_debug('Glamor exploration: collected enough new data, training')
            model = model.train()
            prior_model = prior_model.train()
            inverse = inverse.train()
            args.num_epoch = copy.deepcopy(args.num_train_epoch)
            # updated_model_path = train(model, inverse, buffer, optimizer, skill_lang, args, logdir)
            iterations = train(model, prior_model, inverse, buffer, optimizer, skill_lang, args, logdir, iterations=iterations)

            # update the model weights for the prediction server
            manager.global_dict['model_state_dict'] = state_dict_to_cpu(model.state_dict()) 
            manager.global_dict['prior_model_state_dict'] = state_dict_to_cpu(prior_model.state_dict()) 
            manager.global_dict['inverse_state_dict'] = state_dict_to_cpu(inverse.state_dict())
            # manager.global_dict['model_path'] = updated_model_path
            manager.update_weights()
            manager.run_predictions()
        except KeyboardInterrupt:
            pass

    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true', help='whether or not to pretrain on supervised data')
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not') 
 
    # Generic Parameters for Experiments
    parser.add_argument('--logdir', default='glamor_rl_cachedir', type=str, help='location where log of experiments will be stored')
    parser.add_argument('--rundir', default='runs/glamor_rl_runs')
    parser.add_argument('--exp', default='glamor_debug', type=str, help='name of experiments')
    parser.add_argument('--data_workers', default=4, type=int, help='Number of different data workers to load data in parallel')

    # train
    parser.add_argument('--batch_size', default=16, type=int, help='size of batch of input to use')
    parser.add_argument('--alpha', default=1, type=int, help='relative weight of prior loss vs. inverse model loss')
    parser.add_argument('--num_pretrain_epoch', default=10, type=int, help='number of epochs of training to run')
    parser.add_argument('--num_train_epoch', default=100, type=int, help='number of epochs of training to run')
    parser.add_argument('--num_epoch', default=100, type=int, help='number of epochs of training to run')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for training')
    parser.add_argument('--log_interval', default=10, type=int, help='log outputs every so many batches')
    parser.add_argument('--save_interval', default=100, type=int, help='save outputs every so many batches')
    parser.add_argument('--resume_iter', default=0, type=str, help='iteration to resume training')
    parser.add_argument('--env_episodes_to_add', default=10, type=int, help='number of samples to add to the replay buffer before we train again')

    # model 
    parser.add_argument('--latent_dim', default=512, type=int, help='size of hidden representation')
    parser.add_argument('--max_seq_length', default=5, type=int, help='maximum sequence length')
    parser.add_argument('--n_samples', default=50, type=int, help='maximum sequence length')
    parser.add_argument('--logprob_thresh', default=-10.0, type=float)
    parser.add_argument('--prob_thresh', default=1e-5, type=float)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    main(args)
