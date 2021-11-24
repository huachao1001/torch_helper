
import os
import time
import copy
# from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from torchelper.utils.dist_util import get_bare_model, master_only
from torchelper.models import lr_scheduler as lr_scheduler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
from torchelper.utils.dist_util import get_rank
from torchelper.models.base_model import BaseModel

class ModelBuilder():
    def __init__(self, is_train, ckpt_dir):
        super().__init__()
        self.dataset = None
        self.ckpt_dir = ckpt_dir
        self.is_train = is_train
        self.gpu_id = get_rank()
        if self.gpu_id>=0:
            self.device = torch.device('cuda:'+str(self.gpu_id))
        else:
            self.device = torch.device('cpu')
        self.models = {}
        self.amp_scaler = {}
        # 默认只在GPU训练
        torch.cuda.set_device(self.gpu_id)
   
    def perform_cb(self, event:str, **args):
        for __, model in self.models.items():
            self.get_bare_model(model).perform_cb(event, **args)

    # def get_print_metric(self):
    #     metric_dicts = {}
    #     for __,model in self.models.items():
    #         model = get_bare_model(model)
    #         metric_dic = model.get_metric_dict()
    #         if metric_dic is not None and isinstance(metric_dic, dict):
    #             metric_dicts.update(metric_dic)
    #     msg = []
    #     for k,v in metric_dicts.items():
    #         if v is not None:
    #             msg.append('%s:%.5f' % (k, v))

    #     return ', '.join(msg)
    
    def forward_model(self, model_name, data):
        if model_name in self.amp_scaler.keys():
            with autocast():
                out = self.models[model_name](data)
        else:
            out = self.models[model_name](data)
        return out

    def forward_wrapper(self, epoch, step, data):
        '''一个step执行前向
        :param epoch: int, 当前epoch
        :param step: int, 在当前epoch中的step数
        :param data: 训练集dataset返回的item
        '''
        self.forward(epoch, step, data)
        # if self.is_amp:
        #     # 前向过程开启 autocast
        #         self.forward(epoch, step, data)
        # else:
        #     self.forward(epoch, step, data)

    def on_begin_forward(self, data, epoch, step):
        pass
    def on_end_forward(self,  epoch, step):
        pass
    def on_begin_backward(self,  epoch, step):
        pass
    def on_end_backward(self, epoch, step):
        pass
    
    def backward_wrapper(self):
        #半浮点和全浮点的在各自函数内自定判断
        self.amp_backward()
        self.backward()
 
    def forward(self, epoch, step, data):
        pass
     
    def amp_backward(self):
        '''混合精度训练一个step执行反向
        '''
        for key, model in self.models.items():
            # loss = self.loss_dict.get(key, None) 
            if not key in self.amp_scaler.keys(): #全浮点的走全浮点的反向传播
                continue
            with autocast():
                # loss = self.get_bare_model(model).get_loss()
                loss = self.get_loss(key, self.get_bare_model(model))
            if loss is not None:
                optimizer = self.get_bare_model(model).get_optimizer()
                if optimizer is None:
                    continue
                scaler = self.amp_scaler[key]
                optimizer.zero_grad()
                # Scales loss. 为了梯度放大.
                scaler.scale(loss).backward()
                # scaler.step() 首先把梯度的值unscale回来.
                # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                scaler.step(optimizer)
                # 准备着，看是否要增大scaler
                scaler.update()
        
        # self.step_ema(0.5**(32 / (10 * 1000)))

    def get_loss(self, name, model):
        loss = model.get_loss()
        return loss

    def backward(self):
        '''一个step执行反向
        '''
        for key, model in self.models.items():
            if key in self.amp_scaler.keys(): #半浮点的走半浮点的反向传播
                continue
            # loss = self.loss_dict.get(key, None)
            loss = self.get_loss(key, self.get_bare_model(model))
            loss = loss.float()
            if loss is not None:
                optimizer = self.get_bare_model(model).get_optimizer()
                if optimizer is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

          
    def add_model(self, name:str, model:BaseModel):
        model.builder = self
        self.models[name] = self.model_to_device(model)
        is_half = model.cfg.get('amp', False)
        if is_half:
            self.amp_scaler[name] = GradScaler()
        

    def model_to_device(self, net:nn.Module, find_unused_parameters=False):
        """Model to device. It also warps models with DistributedDataParallel.
        :param net: nn.Module
        """
        net = net.cuda(self.gpu_id)
        net = DistributedDataParallel(
            net, device_ids=[self.gpu_id], find_unused_parameters=find_unused_parameters)
        return net

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_dataset(self):
        return self.dataset

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def set_train(self):
        for name, net in self.models.items():
            net.train()

    def set_eval(self):
        for name, net in self.models.items():
            net.eval()
     

    # @master_only
    # def save_model(self, epoch, max_count=-1, max_time=2*60*60):
    #     '''保存模型, 每隔max_time保存一次，在max_time时间内最多保存max_count个
    #     :param epoch: int, 当前模型epoch
    #     :param max_count: int, 在max_time时间内最多保存数量
    #     :param max_time: int, 秒, 最大时间间隔保存一次
    #     '''
    #     for name, net in self.models.items():
    #         optimizer = self.optimizers.get(name, None)
    #         self.save_model_optimizer(epoch, name, net, optimizer, max_count, max_time)

    #     for name, net in self.ema_models.items():
    #         self.save_model_optimizer(epoch, name+'_ema', net, None, max_count, max_time)
    
    @master_only
    def write_network(self, path):
        '''将网络结构写入文件
        :param path: str, 文件路径
        '''
        with open(path, 'w') as file:
            for net in self.models:
                file.write(str(net))

    
    @master_only    
    def get_metric_dict(self):
        '''用于显示相关测量指标
        :return: 返回字典类型或None
        '''
        return None
    
    @master_only
    def get_scalar_dict(self):
        '''用于显示loss等相关指标
        :return: 返回字典类型或None
        '''
        return None
    
    @master_only
    def get_img_dict(self):
        '''用于显示训练期间中间结果图片
        :return: 返回字典类型或None
        '''
        return None

    @master_only
    def get_audio_dict(self):
        '''用于显示训练期间中间结果音频
        :return: 返回字典类型或None
        '''
        return None