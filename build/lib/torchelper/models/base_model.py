import torch
import torch.nn as nn
from torchelper.utils.dist_util import get_rank


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.builder = None
        self.__optimizer = None
        self.cfg = cfg
        self.init_lr = cfg['init_lr']
        self.callbacks = []
        self.gpu_id = get_rank()
        if self.gpu_id>=0:
            self.device = torch.device('cuda:'+str(self.gpu_id))
        else:
            self.device = torch.device('cpu')
    def get_builder(self):
        return self.builder
        
    def remap_weights_name(self, weights):
        return weights

    def add_cb(self, cb):
        self.callbacks.append(cb)
    
    def perform_cb(self, event:str, **args):
        for cb in self.callbacks:
            evt_func = getattr(cb, event)
            evt_func(self, **args)

    def get_train_params(self):
        '''返回当前model训练参数，如果返回None，则默认为所有训练参数
        '''
        return None
    
    def get_loss(self):
        '''返回当前model loss，如果返回None，则不做梯度更新
        '''
        return None

    def get_optimizer(self):
        if self.__optimizer is None:
            params = self.get_train_params()
            if params is None:
                params = list(self.parameters())
            if len(params)>0:
                self.__optimizer = torch.optim.Adam(params, lr=self.init_lr, betas=(0.5, 0.999))
        return self.__optimizer

