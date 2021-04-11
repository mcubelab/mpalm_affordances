import os, os.path as osp
import time
import threading
from multiprocessing import Process
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import rospkg
rospack = rospkg.RosPack()

class RPOMetricWriter:
    def __init__(self, exp_name, rundir, collect_time=1.0, running_avg_n=10):
        self.exp_name = exp_name
        self.collect_time = collect_time
        self.running_avg_n = running_avg_n
        self.results_dir = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/data/rl_results', self.exp_name)
        if not osp.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.rundir = rundir
 
        # self.collector_thread = threading.Thread(target=self.collect_results)
        # self.collector_thread.daemon = True
        # self.collector_thread.start()
        self.collector_process = Process(
            target=self.collect_results,
            args=(
                self.collect_time,
                self.results_dir,
                self.running_avg_n,
                self.rundir
            ))
        self.collector_process.daemon = True
        # self.collector_process.start()
    
    def collect_results(self, collect_time, results_dir, running_avg_n, rundir):
        start_time = time.time()
        writer = SummaryWriter(rundir)
        samples = 0
        running_mp_success = []

        while True:
            # check if there are any new results
            if time.time() - start_time > collect_time:
                start_time = time.time()
                fnames = os.listdir(results_dir)
                if len(fnames) == 1:
                    if 'master_config.pkl' in fnames[0]:
                        continue
                if len(fnames) > 0:
                    f_ints = []
                    for name in fnames:
                        if name.endswith('.npz'):
                            f_ints.append(int(name.split('.npz')[0]))
                    # f_ints = [int(name.split('.npz')[0]) for name in fnames]
                    # f_ints = sorted(f_ints)
                    f_ints = sorted(f_ints)
                    if samples < f_ints[-1]:
                        samples += 1
                        filename = '%d.npz' % samples
                        data = np.load(osp.join(results_dir, filename), allow_pickle=True)
                        exp_data = data['data'].item()
                        running_mp_success.append(exp_data['mp_success'])
                        
                        # calculate running average of last N
                        if len(running_mp_success) < running_avg_n:
                            running_avg = np.mean(running_mp_success)
                        else:
                            running_avg = np.mean(running_mp_success[-running_avg_n:])
                        
                        # write to tb
                        writer.add_scalar('planning/mp_success', running_avg, samples)
            else:
                time.sleep(0.1)
