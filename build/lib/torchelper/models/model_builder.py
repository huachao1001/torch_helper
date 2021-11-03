
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

from torchelper.models.base_model import BaseModel

class ModelBuilder():
    def __init__(self, is_train, ckpt_dir, gpu_id, is_amp):
        super().__init__()
        self.dataset = None
        self.ckpt_dir = ckpt_dir
        self.is_train = is_train
        self.is_amp = is_amp
        self.gpu_id = gpu_id
        if gpu_id>=0:
            self.device = torch.device('cuda:'+str(gpu_id))
        else:
            self.device = torch.device('cpu')
        # self.is_dist = is_dist
        self.models = {}
        # self.schedulers = {}
        # self.optimizers = {}
        # self.loss_funcs = {}
        # self.loss_dict = {}
        self.amp_scaler = {}
        # self.ema_models = {}
        # self.ema_flags = {}   # 用于记录ema模型是否是第一次做平滑
        # if not os.path.exists(os.path.join(self.ckpt_dir, "imgs")):
        #     os.makedirs(os.path.join(self.ckpt_dir, "imgs"))
        # 默认只在GPU训练
        torch.cuda.set_device(gpu_id)
        # self.device = torch.device('cuda')
        self.last_interval_save_time = -1
        self.pth_list = []
    
    def perform_cb(self, event:str, **args):
        for __, model in self.models.items():
            self.get_bare_model(model).perform_cb(event, **args)

    def get_print_metric(self):
        metric_dicts = {}
        for __,model in self.models.items():
            model = get_bare_model(model)
            metric_dic = model.get_metric_dict()
            if metric_dic is not None and isinstance(metric_dic, dict):
                metric_dicts.update(metric_dic)
        msg = []
        for k,v in metric_dicts.items():
            if v is not None:
                msg.append('%s:%.5f' % (k, v))

        return ', '.join(msg)

    def forward_wrapper(self, epoch, step, data):
        '''一个step执行前向
        :param epoch: int, 当前epoch
        :param step: int, 在当前epoch中的step数
        :param data: 训练集dataset返回的item
        '''
        if self.is_amp:
            # 前向过程开启 autocast
            with autocast():
                self.forward(epoch, step, data)
        else:
            self.forward(epoch, step, data)
    def on_begin_forward(self, data, epoch, step):
        pass
    def on_end_forward(self,  epoch, step):
        pass
    def on_begin_backward(self,  epoch, step):
        pass
    def on_end_backward(self, epoch, step):
        pass
    # def criterion_wrapper(self):
    #     if self.is_amp:
    #         # 前向过程开启 autocast
    #         with autocast():
    #             self.criterion()
    #     else:
    #         self.criterion()
    
    def backward_wrapper(self):
        if self.is_amp:
            self.amp_backward()
        else:
            self.backward()
 
    def forward(self, epoch, step, data):
        pass
    
    #用于在tensorboard显示数据
    def get_vis_dict(self):
        #返回格式：{'name':{'type':'image|scalar', 'val':data}, ...}
        return None
    # def criterion(self):
    #     '''计算loss
    #     :param epoch: int, 当前epoch
    #     :param step: int, 在当前epoch中的step数
    #     :param data: 训练集dataset返回的item
    #     :return: 子类一般返回forward结果
    #     '''
    #     for key in self.models.keys():
    #         loss_func = self.loss_funcs[key]
    #         if loss_func is not None:
    #             self.loss_dict[key] = loss_func()

       
    def amp_backward(self):
        '''混合精度训练一个step执行反向
        '''
        for key, model in self.models.items():
            # loss = self.loss_dict.get(key, None)
            
            with autocast():
                loss = self.get_bare_model(model).get_loss()
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
            # loss = self.loss_dict.get(key, None)
            loss = self.get_loss(key, self.get_bare_model(model))
            if loss is not None:
                optimizer = self.get_bare_model(model).get_optimizer()
                if optimizer is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # loss_func = self.loss_funcs[key]
            # if loss_func is not None:
            #     loss = loss_func()
            # if loss is not None:
            #     optimizer = self.optimizers[key]
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
        
        # self.step_ema(0.5**(32 / (10 * 1000)))

    # @abstractmethod
    # def validate(self, epoch, data):
    #     '''跑验证集
    #     :param epoch: int, 当前epoch
    #     :param step: int, 在当前epoch中的step数
    #     :param data: 验证集dataset返回的item
    #     :return: 返回None或dict，其中dict为各项测试指标键值对
    #     '''
    # def before_validate(self, epoch):
    #     pass
    
    # def after_validate(self, epoch):
    #     return {}
    
    # def step_ema(self, decay):
    #     for name, net_ema in self.ema_models.items():
    #         if self.ema_flags[name]:
    #             self.model_ema(self.models[name], net_ema, 0)
    #         else:
    #             self.model_ema(self.models[name], net_ema, decay)
    #         self.ema_flags[name] = False    # 标记当前ema模型已做平滑

    def add_model(self, name:str, model:BaseModel):
        
        self.models[name] = self.model_to_device(model)
        if self.is_amp:
            self.amp_scaler[name] = GradScaler()
       
    # def add_model(self, name, cls_str, init_lr, opt_type, loss_func, model_cfg, lr_scheduler):
    #     model = self.create_net(cls_str, model_cfg)
    #     model = self.model_to_device(model)
    #     optimizer = self.create_optimizer(model, opt_type, init_lr)
    #     self.models[name] = model
    #     self.optimizers[name] = optimizer
    #     self.loss_funcs[name] = loss_func
    #     self.schedulers[name] = lr_scheduler
    #     if self.is_amp:
    #         self.amp_scaler[name] = GradScaler()

    # def create_net(self, class_str, cfg):
    #     '''动态创建模型
    #     :param class_str: str, 类名称
    #     :return: Model, class_str对应的模型类
    #     '''
    #     arr = class_str.split('.')
    #     g_module = __import__('.'.join(arr[0:-1]) , fromlist = True)
    #     g_cls = getattr(g_module, arr[-1])
    #     return g_cls(cfg)

    # def add_ema_model(self, name):
    #     model = self.get_bare_model(self.models[name])
    #     ema_net = copy.deepcopy(model)
    #     ema_net = self.model_to_device(ema_net)
    #     # ema_net = ema_net.cuda(self.gpu_id)
    #     ema_net.eval()
    #     self.ema_models[name] = ema_net
    #     self.ema_flags[name] = True

    def model_to_device(self, net:nn.Module, find_unused_parameters=False):
        """Model to device. It also warps models with DistributedDataParallel.
        :param net: nn.Module
        """
        net = net.cuda(self.gpu_id)
        # net = DataParallel(net, device_ids=self.gpu_ids)
        # if self.is_dist:
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

    # def create_optimizer(self, net, opt_type, init_lr, beta1=0.5, beta2=0.999):
    #     '''创建优化器
    #     :param net: nn.Module
    #     :param init_lr: 初始学习率
    #     :return: 返回优化器对象
    #     '''
    #     if opt_type == 'adam':
    #         optimizer = torch.optim.Adam(list(net.parameters()), lr=init_lr, betas=(beta1, beta2))
    #     else:
    #         raise f'optimizer {opt_type} is not supperted yet.'
    #     return optimizer

    # def model_ema(self, net, net_ema, decay=0.999):
    #     '''对模型参数做平滑
    #     :param net: 训练模型
    #     :param net_ema: 平滑后的模型
    #     '''
    #     net_g_params = dict(net.named_parameters())
    #     net_g_ema_params = dict(net_ema.named_parameters())
    #     for k in net_g_ema_params.keys():
    #         net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def save_model_optimizer(self, epoch, name, model, optimizer, max_count=-1, max_time=2*60*60):
        pass
        # #1. save optimizer
        # if optimizer is not None:
        #     save_opt_name = "%s_optimizer_%s.pth" % (epoch, name)
        #     save_opt_path = os.path.join(self.ckpt_dir, save_opt_name)
        #     torch.save(optimizer.state_dict(), save_opt_path)
        #     print('save:', save_opt_path)

        # #2. save weights
        # save_weight_filename = "%s_weights_%s.pth" % (epoch, name)
        # save_weight_path = os.path.join(self.ckpt_dir, save_weight_filename)
        # torch.save(self.get_bare_model(model).state_dict(), save_weight_path)
        # #2. save ema weights
        # if self.ema_models.get(name, None) is not None:
        #     save_ema_filename = "%s_weights_%s.pth" % (epoch, name+'_ema')
        #     save_ema_path = os.path.join(self.ckpt_dir, save_ema_filename)
        #     torch.save(self.get_bare_model(self.ema_models[name]).state_dict(), save_ema_path)

        # if max_count<=0:
        #     return
        # #3. clear old
        # # 每间隔max_time时间保存一次
        # if time.time() - self.last_interval_save_time > max_time:
        #     self.last_interval_save_time = time.time()
        # else:
        #     # print('before:', self.pth_list)
        #     if epoch not in self.pth_list:
        #         self.pth_list.append(epoch)
        #     # print('before:', self.pth_list)
        #     while len(self.pth_list) > max_count and max_count > 1:
        #         for name in os.listdir(self.ckpt_dir):
        #             if name.endswith('.pth'):
        #                 is_opt = name.startswith('%s_optimizer_'%self.pth_list[0])
        #                 is_weight = name.startswith('%s_weights_'%self.pth_list[0])
        #                 if is_opt or is_weight:
        #                     path = os.path.join(self.ckpt_dir, name)
        #                     os.remove(path)
        #         del self.pth_list[0]

    def load_model_optimizer(self, epoch, name, model, optimizer):
        pass
        # 1. load weights
        # save_filename = "%s_weights_%s.pth" % (epoch, name)
        # save_path = os.path.join(self.ckpt_dir, save_filename)
        # if os.path.exists(save_path):
        #     # n_w = {}
        #     weights = torch.load(save_path, map_location=lambda storage, loc: storage)
        #     # for k,v in weights.items():
        #     #     n_w['module.'+k] = v
        #     self.get_bare_model(model).load_state_dict(weights, strict=False)
        #     print("success load model:"+ save_path)
        # else:
        #     print("%s not exists yet!" % save_path)
        #     return False

        # # 2. load ema
        # ema_model = self.ema_models.get(name, None)
        # if ema_model is not None:
        #     ema_file = "%s_weights_%s_ema.pth" % (epoch, name)
        #     ema_path = os.path.join(self.ckpt_dir, ema_file)
        #     if os.path.exists(ema_path):
        #         ema_weights = torch.load(ema_path, map_location=lambda storage, loc: storage)
        #         self.get_bare_model(ema_model).load_state_dict(ema_weights)
        #         print("success load model:"+ ema_path)
        #     else:
        #         self.get_bare_model(ema_model).load_state_dict(weights)
        #         print("%s not exists yet! load weights from %s!" % (ema_path, save_path))
        #     self.ema_flags[name] = False

        # # 3. load optimizer
        # if optimizer is not None:
        #     save_filename = "%s_optimizer_%s.pth" % (epoch, name)
        #     save_path = os.path.join(self.ckpt_dir, save_filename)
        #     if not os.path.isfile(save_path):
        #         print("%s not exists yet!" % save_path)
        #         return False
        #     else:
        #         # weights = torch.load(save_path, map_location=lambda storage, loc: storage)
        #         # optimizer.load_state_dict(weights)
        #         print("success load optimizer:", save_path)

    def load_model(self, epoch):
        '''加载预训练模型
        :param epoch: int, 预训练模型epoch
        '''
        # for name, model in self.models.items():
        #     # if 'eye' in name or 'mouth' in name:
        #     #     continue
        #     self.load_model_optimizer(epoch, name, model, self.optimizers[name])

        # if self.is_dist:
        # 多卡同步, 确保没块卡已经加载好模型参数
        torch.distributed.barrier()

    @master_only
    def save_model(self, epoch, max_count=-1, max_time=2*60*60):
        '''保存模型, 每隔max_time保存一次，在max_time时间内最多保存max_count个
        :param epoch: int, 当前模型epoch
        :param max_count: int, 在max_time时间内最多保存数量
        :param max_time: int, 秒, 最大时间间隔保存一次
        '''
        for name, net in self.models.items():
            optimizer = self.optimizers.get(name, None)
            self.save_model_optimizer(epoch, name, net, optimizer, max_count, max_time)

        for name, net in self.ema_models.items():
            self.save_model_optimizer(epoch, name+'_ema', net, None, max_count, max_time)
    @master_only
    def write_network(self, path):
        '''将网络结构写入文件
        :param path: str, 文件路径
        '''
        with open(path, 'w') as file:
            for net in self.models:
                file.write(str(net))

    def update_learning_rate(self, epoch, warmup_epoch=-1):
        '''更新学习率
        '''
        pass
        # for name, scheduler in self.schedulers.items():
        #     optimizer = self.optimizers[name]
        #     lr = scheduler.get_lr(epoch)
        #     if epoch < warmup_epoch:
        #         init_lr = scheduler.init_lr
        #         lr = init_lr / warmup_epoch * epoch
            
        #     for param_group in optimizer.param_groups :
        #         param_group['lr'] = lr