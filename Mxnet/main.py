# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:30:55 2021


@author: Yiji Zhao

"""

import sys
import os
import math
import random
import argparse
import platform
import numpy as np
from easydict import EasyDict as edict
from timeit import default_timer as timer

import mxnet as mx
from mxnet import nd
from mxnet import gpu
from mxnet import cpu
from mxnet import init
from mxnet import gluon
from mxnet import autograd

from STPGCN import Model

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
os.environ["MXNET_CUDA_LIB_CHECKING"] = "0"

def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(array)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))

def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))

class Metric(object):

    def __init__(self, num_prediction):
        self.time_start = timer()
        self.num_prediction = num_prediction
        self.best_metrics  = {'mae':np.inf, 'rmse':np.inf, 'mape':np.inf, 'epoch': np.inf}
        self.step_metrics_epoch = {'mae':{}, 'rmse':{}, 'mape':{}}

    def update_metrics(self, y_true, y_pred):
        self.metrics = {'mae':0.0, 'rmse':0.0, 'mape':0.0, 'time':0.0}
        self.metrics['mae'], self.metrics['rmse'], self.metrics['mape'], mse = self.get_metric(y_true, y_pred)
        self.metrics['time'] = time_to_str((timer() - self.time_start))

    def update_step_metrics(self, y_true, y_pred, epoch=0):
        idx_lst=['mae','rmse','mape']
        
        metrics = {}
        for i in idx_lst:
            metrics[i] = [0.0]*(self.num_prediction+1)

        metrics['mae'][-1], metrics['rmse'][-1], metrics['mape'][-1], _ = self.get_metric(y_true, y_pred)

        for t in range(self.num_prediction):
            true, pred = y_true[:,t,:,:], y_pred[:,t,:,:]
            metrics['mae'][t], metrics['rmse'][t], metrics['mape'][t], _ = self.get_metric(true, pred)
    
        for i in idx_lst:
            self.step_metrics_epoch[i][epoch] = metrics[i]

    def update_best_metrics(self, epoch=0):
        self.best_metrics['mae'],  mae_state  = self.get_best_metric(self.best_metrics['mae'],  self.metrics['mae'])
        self.best_metrics['rmse'], rmse_state = self.get_best_metric(self.best_metrics['rmse'], self.metrics['rmse'])
        self.best_metrics['mape'], mape_state = self.get_best_metric(self.best_metrics['mape'], self.metrics['mape'])
 
        if mae_state:
            self.best_metrics['epoch'] = int(epoch)
        
    @staticmethod
    def get_metric(y_true, y_pred):
        mae  = masked_mae_np(y_true, y_pred, 0)
        mse  = masked_mse_np(y_true, y_pred, 0)
        mape = masked_mape_np(y_true, y_pred, 0)
        rmse = mse ** 0.5
        return mae, rmse, mape, mse
        
    @staticmethod
    def get_best_metric(best, candidate):
        state = False
        if candidate < best: 
            best = candidate
            state = True
        return best, state
        
    def __str__(self):
        """For print"""
        return f"{self.metrics['mae']:<7.2f}{self.metrics['rmse']:<7.2f}{self.metrics['mape']:<7.2f} | {self.best_metrics['epoch']+1:<4}"

    def best_str(self):
        """For save"""
        return f"{self.best_metrics['epoch']},{self.best_metrics['mae']:.2f},{self.best_metrics['rmse']:.2f},{self.best_metrics['mape']:.2f}"

    def multi_step_str(self, obj='rmse', sep=',', epoch=0):
        """For print or save""" #"{i+1}:{x:<7.2f}"
        return sep.join([f"{x:.2f}" if sep ==',' else f"{x:<6.2f}" for i,x in enumerate(self.step_metrics_epoch[obj][epoch])])

    def log_lst(self,epoch=None,sep=','):
        message_lst = []
        index = ['mae','rmse','mape']
        
        for i in index:
            message_lst.append(f"{i},{self.multi_step_str(obj=i, sep=sep, epoch=epoch)}")
        return message_lst
        
# print logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=True, is_file=True):
        if '\r' in message: is_file=False

        if is_terminal:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file:
            self.file.write(message)
            self.file.flush()

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError
        
def search_recent_data(train, label_start_idx, points_per_hour, num_prediction):
    if label_start_idx + num_prediction > len(train): return None
    start_idx, end_idx = label_start_idx - num_prediction, label_start_idx - num_prediction + num_prediction    
    if start_idx < 0 or end_idx < 0: return None
    return (start_idx, end_idx), (label_start_idx, label_start_idx + num_prediction)
        
def search_multihop_neighbor(adj, hops = 5):
    node_cnt = adj.shape[0]
    hop_arr = np.zeros((adj.shape[0], adj.shape[0]))
    for h_idx in range(node_cnt):  # refer node idx(n)
        tmp_h_node, tmp_neibor_step = [h_idx], [h_idx]  # save spatial corr node  # 0 step(self) first
        hop_arr[h_idx, :] = -1  # if the value exceed maximum hop, it is set to (hops + 1)
        hop_arr[h_idx, h_idx] = 0  # at begin, the hop of self->self is set to 0
        for hop_idx in range(hops):  # how many spatial steps
            tmp_step_node = []  # neighbor nodes in the previous k step
            tmp_step_node_kth = []  # neighbor nodes in the kth step
            for tmp_nei_node in tmp_neibor_step:
                tmp_neibor_step = list((np.argwhere(adj[tmp_nei_node] == 1).flatten()))  # find the one step neighbor first
                tmp_step_node += tmp_neibor_step
                tmp_step_node_kth += set(tmp_step_node) - set(tmp_h_node)  # the nodes that have appeared in the first k-1 step are no longer needed
                tmp_h_node += tmp_neibor_step
            tmp_neibor_step = tmp_step_node_kth.copy()
            all_spatial_node = list(set(tmp_neibor_step))  # the all spatial node in kth step
            hop_arr[h_idx, all_spatial_node] = hop_idx + 1
    return hop_arr[:, :, np.newaxis]

class CleanDataset():
    def __init__(self, config):
        
        self.data_name      = config.data.name
        self.feature_file   = config.data.feature_file
        self.val_start_idx  = config.data.val_start_idx
        self.alpha          = config.model.alpha
        self.t_size         = config.model.t_size
        
        self.adj = np.load(config.data.spatial)
        self.label, self.feature = self.read_data()
        self.spatial_distance = search_multihop_neighbor(self.adj, hops=self.alpha)
        self.range_mask = self.interaction_range_mask(hops=self.alpha, t_size=self.t_size)

    def read_data(self):
        if 'PEMS' in self.data_name:
            data = np.expand_dims(np.load(self.feature_file)[:,:,0],-1)
        else:
            data = np.load(self.feature_file)
        return data.astype('float32'), self.normalization(data).astype('float32')
        
    def normalization(self, feature):
        train = feature[:self.val_start_idx]
        if 'Metro' in self.data_name:
            idx_lst = [i for i in range(train.shape[0]) if i % (24*6) >= 7*6 - 12]
            train = train[idx_lst]

        mean = np.mean(train)  
        std  = np.std(train)
        return (feature - mean) / std

    def interaction_range_mask(self, hops = 2, t_size=3):
        hop_arr = self.spatial_distance
        hop_arr[hop_arr != -1] = 1
        hop_arr[hop_arr == -1] = 0
        return np.concatenate([hop_arr.squeeze()]*t_size, axis=-1) # V,tV


class TrafficDataset(gluon.data.Dataset):
    def __init__(self, clean_data, data_range, config):
        self.T   = config.model.T
        self.V   = config.model.V
        self.ctx = config.model.ctx
        self.points_per_hour = config.data.points_per_hour
        self.num_prediction  = config.model.num_prediction
        self.data_range = data_range
        self.data_name  = clean_data.data_name
        
        self.label   = nd.array(clean_data.label,dtype='float32',ctx=self.ctx)
        self.feature = nd.array(clean_data.feature,dtype='float32',ctx=self.ctx) #(T, V, D)
        
        # Prepare samples
        self.idx_lst = self.get_idx_lst()
        print('sample:',len(self.idx_lst))
          
    def __getitem__(self, index):

        recent_idx  = self.idx_lst[index]

        start,end = recent_idx[1][0],recent_idx[1][1]
        label = self.label[start:end]

        start,end = recent_idx[0][0],recent_idx[0][1]
        node_feature     = self.feature[start:end]
        pos_w,pos_d = self.get_time_pos(start)
        
        return label, node_feature, pos_w, pos_d
    
    def __len__(self):
        return len(self.idx_lst)

    def get_time_pos(self, idx):
        idx = np.array(range(self.T)) + idx
        pos_w = (idx // (self.points_per_hour * 24)) % 7 #day of week
        pos_d = idx % (self.points_per_hour * 24)        #time of day
        return pos_w, pos_d

    def get_idx_lst(self):
        idx_lst = []
        start = self.data_range[0]
        end   = self.data_range[1] if self.data_range[1]!=-1 else self.feature.shape[0]
        
        for label_start_idx in range(start,end):
            # only 6:00-24:00 for Metro data
            if 'Metro' in self.data_name:  
                if label_start_idx % (24 * 6) < (7*6):
                    continue
                if label_start_idx % (24 * 6) > (24*6) - self.num_prediction:
                    continue
            recent = search_recent_data(self.feature, label_start_idx, self.points_per_hour, self.num_prediction)  # recent data

            if recent:
                idx_lst.append(recent) 
        return idx_lst 


    
#################################################
def train(model, data_loader, trainer, loss_function, epoch, metric, config):

    y_pred, y_true, time_lst = [],[],[]
    for i, (target,feature, pos_w, pos_d) in enumerate(data_loader):

        time_start = timer()
        with autograd.record():
            output = model(feature, pos_w, pos_d)
            l = loss_function(output, target)
        l.backward(retain_graph=True)
        trainer.step(target.shape[0])
        time_lst.append((timer() - time_start))
                        
        y_true.append(target.asnumpy())
        y_pred.append(output.asnumpy())

        if i == 0 and epoch==0:
            num_of_parameters = 0
            for param_name, param_value in model.collect_params().items():
                num_of_parameters += np.prod(param_value.shape)
            print('\nNum_of_parameters:', num_of_parameters)
            print("Epoch | Tra: MAE RMSE MAPE Time | Val: MAE RMSE MAPE Time | Tes:  MAE RMSE MAPE Time")
                
        message = f"{i/len(data_loader)+epoch:6.1f} Time:{np.sum(time_lst):.1f}s"
        print('\r'+message , end='', flush=True)
    
    y_true = np.concatenate(y_true,axis=0)
    y_pred = np.concatenate(y_pred,axis=0)
    
    time_cost = np.sum(time_lst)
    metric.update_metrics(y_true,y_pred)
    metric.update_best_metrics(epoch=epoch)
    
    message = f"{epoch+1:<3} | {metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}{metric.metrics['mape']:<7.2f}{time_cost:<5.2f}s"
    print('\r'+message , end='', flush=False)
    
    message = f"{'Train':5}{epoch+1:6.1f} | {str(metric)}{time_cost:.1f}s"
    config.logger.write('\n'+message+'\n',is_terminal=False)
    
    return metric

def evals(model, data_loader, epoch, metric, config, mode='Test', end=''):

    if mode == 'Test': end='\n'
    y_pred, y_true, time_lst = [],[],[]
    for i, (target,feature, pos_w, pos_d) in enumerate(data_loader):
        
        time_start = timer()
        output = model(feature, pos_w, pos_d)
        time_lst.append((timer() - time_start))
        
        y_true.append(target.asnumpy())
        y_pred.append(output.asnumpy())
        
    y_true = np.concatenate(y_true,axis=0)
    y_pred = np.concatenate(y_pred,axis=0)    
        
    time_cost = np.sum(time_lst)
    metric.update_metrics(y_true,y_pred)
    metric.update_best_metrics(epoch=epoch)
    
    message = f" | {metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}{metric.metrics['mape']:<7.2f}{time_cost:<5.2f}s" 
    print(message , end=end, flush=False)
    message = f"{mode:5}{epoch+1:6.1f} | {str(metric)}{time_cost:.1f}s"
    config.logger.write(message+'\n',is_terminal=False)

    if mode != 'Val':
        metric.update_step_metrics(y_true,y_pred,epoch=epoch)
        for i,m in enumerate(metric.log_lst(epoch=epoch,sep=',')):
            config.logger.write(m+'\n',is_terminal=False)
            
    return metric


#######################################
class MyInit(init.Initializer):
    normal   = init.Normal(0.1)
    constant = init.Constant(1)
    xavier   = init.Xavier()
    uniform  = init.Uniform()
    def _init_weight(self, name, data):
        if 'mu' in name:
            self.normal._init_weight(name, data)
            print('Init', name, data.shape, 'with Normal')
        elif 'sigma' in name:
            self.constant._init_weight(name, data)
            print('Init', name, data.shape, 'with Constant')
        elif len(data.shape) < 2:
            self.uniform._init_weight(name, data)
            print('Init', name, data.shape, 'with Uniform')
        else:
            self.xavier._init_weight(name, data)
            print('Init', name, data.shape, 'with Xavier')

def main(config):

    model = Model(config=config.model)

    print('Initialize model ...')
    if(config.model.start_epochs>0):
        print('read params:',config.model.init_params)
        model.load_parameters(filename=config.model.init_params, ctx=config.model.ctx)
    else:
        model.initialize(ctx=config.model.ctx, init=MyInit(), force_reinit = True)

    model.hybridize()

    loss_function = gluon.loss.L1Loss()
    trainer = gluon.Trainer(model.collect_params(), config.model.optimizer, {'learning_rate':config.model.learning_rate})

    # Traning and testing
    config.model.logger.open(config.model.log_file,mode="a")
    config.model.logger.write(f"Workspace:{config.model.workname}\nModel:{config.model.name}\n\n")
    config.model.logger.write(f"{'Type':^5}{'Epoch':^5} | {'MAE':^7}{'RMSE':^7}{'MAPE':^7} | Best-Epoch-of-MAE TimeCost\n",is_terminal=False)

    metrics_tra = Metric(num_prediction=config.model.num_prediction)
    metrics_val   = Metric(num_prediction=config.model.num_prediction)
    metrics_tes  = Metric(num_prediction=config.model.num_prediction)
    
    for epoch in range(config.model.start_epochs, config.model.end_epochs):
        # Training
        train(model, config.data.train_loader, trainer, loss_function, epoch, metrics_tra, config=config.model)
        # Validation
        evals(model, config.data.val_loader, epoch, metrics_val, config=config.model, mode='Val')            
        
        if metrics_val.best_metrics['epoch'] == epoch:
            if epoch>30:
                params_filename = config.PATH_MOD+f"BestVal-{config.rid}_{config.model.name}.params"
                model.save_parameters(params_filename)
            # Testing
            evals(model, config.data.test_loader, epoch, metrics_tes, config=config.model, mode='Test')
        else:
            print()
            
        if epoch-metrics_val.best_metrics['epoch']>config.model.early_stop: # 早停阈值
            break

    config.model.logger.write('\n'+('-'*20)+'\n',is_terminal=True)
    config.model.logger.write(str_config(config)+'\n',is_terminal=True)
    config.model.logger.write(('-'*20)+'\n',is_terminal=True)
    
    # Final records
    best_val_epoch = metrics_val.best_metrics['epoch']
    config.model.logger.write(f"Best Validation MAE at Epoch:{best_val_epoch+1:^3} | Corresponding Testing MAE:{metrics_tes.step_metrics_epoch['mae'][best_val_epoch][-1]:.2f} RMSE:{metrics_tes.step_metrics_epoch['rmse'][best_val_epoch][-1]:.2f} MAPE:{metrics_tes.step_metrics_epoch['mape'][best_val_epoch][-1]:.2f}\n",is_terminal=True)
    config.model.logger.write(f"{'-'*20}\nMetric,{','.join(['step'+str(i+1) for i in range(config.model.num_prediction)])},Total\n",is_terminal=True)

    
    message_lst = metrics_tes.log_lst(epoch=best_val_epoch,sep=',')
    
    for i,m in enumerate(message_lst):
        if i%6==0: print(('-'*20))
        config.model.logger.write(m+'\n',is_terminal=True)

    with open(config.fsummary,mode='a') as f:
        for m in message_lst:
            f.write(f"{config.rid},{m}\n")
        f.close()

    return metrics_val,metrics_tes


def default_config(data='PEMS08',workname='record'):

    config = edict()
    config.PATH_LOG  = '%s/Log/'%workname
    config.PATH_NPY  = '%s/Forecasting/'%workname
    config.PATH_MOD  = '%s/Model/'%workname
    config.rid    = 0
    
    # Data Config
    config.data = edict()
    config.data.name = data
    config.data.path = 'dataset/'

    config.data.feature_file = config.data.path+config.data.name+'/flow.npy'
    config.data.spatial = config.data.path+config.data.name+'/adj.npy'

    if config.data.name=='MetroHZ': 
        config.data.num_features    = 2
        config.data.num_vertices    = 80
        config.data.points_per_hour = 6
        config.data.val_start_idx   = int(3744 * 0.6)
        config.data.test_start_idx  = int(3744 * 0.8)

    if config.data.name == 'BikeNYC':
        config.data.num_features    = 2
        config.data.num_vertices    = 128
        config.data.points_per_hour = 1
        config.data.val_start_idx   = int(4392 * 0.6)
        config.data.test_start_idx  = int(4392 * 0.8)

    if config.data.name == 'PEMS03':
        config.data.num_features    = 1
        config.data.num_vertices    = 358
        config.data.points_per_hour = 12
        config.data.val_start_idx   = int(26208 * 0.6)
        config.data.test_start_idx  = int(26208 * 0.8)

    if config.data.name == 'PEMS04':
        config.data.num_features    = 1
        config.data.num_vertices    = 307
        config.data.points_per_hour = 12
        config.data.val_start_idx   = int(16992 * 0.6)
        config.data.test_start_idx  = int(16992 * 0.8)
    
    if config.data.name == 'PEMS07':
        config.data.num_features    = 1
        config.data.num_vertices    = 883
        config.data.points_per_hour = 12
        config.data.val_start_idx   = int(28224 * 0.6)
        config.data.test_start_idx  = int(28224 * 0.8)

    if config.data.name == 'PEMS08':
        config.data.num_features    = 1
        config.data.num_vertices    = 170
        config.data.points_per_hour = 12
        config.data.val_start_idx   = int(17856 * 0.6)
        config.data.test_start_idx  = int(17856 * 0.8)

    # Model Config
    config.model = edict()
    config.model.workname   = workname
    config.model.logger     = Logger()
    config.model.gpu_id     = 0
    config.model.optimizer  = "adam"
    config.model.learning_rate = 0.002
    config.model.wd         = 1e-5
    config.model.early_stop = 150

    config.model.start_epochs  = 0
    config.model.end_epochs    = 200
    config.model.init_params   = None
    config.model.ctx = gpu(config.model.gpu_id)
        
    config.model.num_prediction  = 12    
    config.model.V = config.data.num_vertices
    config.model.T = config.model.num_prediction
    config.model.num_features = config.data.num_features
    config.model.day_len = config.data.points_per_hour * 24
    
    
    if not os.path.exists(config.PATH_LOG):
        os.makedirs(config.PATH_LOG)

    if not os.path.exists(config.PATH_NPY):
        os.makedirs(config.PATH_NPY)

    if not os.path.exists(config.PATH_MOD):
        os.makedirs(config.PATH_MOD)

    return config

def str_config(config):
    str_cfg = (f"Data:{config.data.name} Spatio-Temporal Range:a{config.model.alpha}b{config.model.beta} History:{config.model.T} Predict:{config.model.num_prediction} Batch:{config.model.batch_size} lr:{config.model.learning_rate}\n"
               f"Layer:{config.model.L} Dim:{config.model.C} Random Seed:{config.seed}  Dim of embedding:{config.model.d}"
               )
    return str_cfg


def run(args):
    
    config = default_config(data=args.data,workname=args.workname)
    
    config.seed = args.seed
    config.rid  = args.rid
    config.model.batch_size = args.batch
    config.model.early_stop = args.early_stop
    config.model.end_epochs = args.max_epoch

    config.model.L      = args.L # Number of layer
    config.model.C      = args.C # Dimension of node features
    config.model.d      = args.d # Dimension of position embedding
    config.model.alpha  = args.a # Alpha
    config.model.beta   = args.b # Beta
    config.model.t_size = args.b+1

    
    config.model.DEBUG = args.DEBUG

    config.model.name = (f"{config.data.name}_a{config.model.alpha}b{config.model.beta}"
                         f"_L{config.model.L}_d{config.model.d}_C{config.model.C}_h{config.model.T}p{config.model.num_prediction}"
                         f"_b{config.model.batch_size}_lr{math.ceil(config.model.learning_rate*1000)}e-3")

    config.fsummary  = f'{config.PATH_LOG}{config.model.name}.csv'
    config.model.log_file = f"{config.PATH_LOG}{config.rid}_{config.model.name}.log"
    config.model.file_forecast  = f"{config.PATH_NPY}{config.rid}_{config.model.name}.npy"

    #  data pre-processing
    print('\n1. data pre-processing ...')
    clean_data = CleanDataset(config)
    config.model.range_mask = clean_data.range_mask
    config.model.spatial_distance = clean_data.spatial_distance
    
    # Dataset
    print('2. data loader ...')
    train_data = TrafficDataset(clean_data, (0+config.model.num_prediction,config.data.val_start_idx-config.model.num_prediction + 1), config)
    val_data   = TrafficDataset(clean_data, (config.data.val_start_idx + config.model.num_prediction,config.data.test_start_idx-config.model.num_prediction + 1), config)
    test_data  = TrafficDataset(clean_data, (config.data.test_start_idx + config.model.num_prediction,-1), config)
    
    # Improve GPU utilization. If the inference time is tested, batch_mul should be set to 1.
    batch_mul = 2 if '07' in config.data.name else 4
    config.data.train_loader = gluon.data.DataLoader(train_data,batch_size=config.model.batch_size,shuffle=True)
    config.data.val_loader   = gluon.data.DataLoader(val_data,  batch_size=config.model.batch_size*batch_mul,shuffle=False)
    config.data.test_loader  = gluon.data.DataLoader(test_data, batch_size=config.model.batch_size*batch_mul,shuffle=False)
    
    main(config)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rid",        type=int, default=1)
    parser.add_argument("--data",       type=str, default='PEMS08')
    parser.add_argument("--seed",       type=int, default=1)
    parser.add_argument("--batch",      type=int, default=32)
    parser.add_argument("--L",          type=int, default=3)
    parser.add_argument("--a",          type=int, default=4)
    parser.add_argument("--b",          type=int, default=2)
    parser.add_argument("--d",          type=int, default=8)
    parser.add_argument("--C",          type=int, default=64)
    parser.add_argument("--max_epoch",  type=int, default=200)
    parser.add_argument("--early_stop", type=int, default=150)
    parser.add_argument("--workname",   type=str, default='STPGCN')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    run(args)
    

    # cmd: python main.py --rid=1 --seed=1 --L=3 --a=4 --b=2 --d=8 --data=PEMS08 --batch=32 --C=64 --workname=STPGCN-PEMS08
    
    
