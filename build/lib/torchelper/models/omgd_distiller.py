import torch
import math
import torch.nn as nn
import itertools
from torchelper import ModelGroup, SSIMNet
from ..archs.vgg_19 import VGG19_torch
import torch.nn.functional as F
from torch.nn.parallel import gather, parallel_apply, replicate
from torch.cuda.amp import GradScaler

class OMGDDistiller(ModelGroup):

    def __init__(self, cfg, gpu_id, is_train, is_dist, is_amp):
        super().__init__(is_train, cfg['ckpt_dir'], gpu_id, is_dist, is_amp)
        self.connectors = []
        self.teacher_w = None
        self.student = None
        self.vgg = VGG19_torch().cuda(gpu_id)
    
 
    def add_student(self, teacher_w, name, cls_str, init_lr, loss_func, model_cfg, lr_scheduler):
        self.teacher_w = self.get_bare_model(teacher_w)
        self.student = self.create_net(cls_str, model_cfg)

        teacher_chans = self.teacher_w.get_mid_features()
        student_chans = self.student.get_mid_features()
        params = [self.student.parameters()]
        for t_chan, s_chan in zip(teacher_chans, student_chans):
            module = self.build_feature_connector(t_chan, s_chan)
            params.append(module.parameters())
            self.connectors.append(module.cuda(self.gpu_id))
        optimizer = torch.optim.Adam(itertools.chain(*params), lr=init_lr, betas=(0.5, 0.999))
        self.optimizers[name] = optimizer
        self.loss_funcs[name] = loss_func
        self.schedulers[name] = lr_scheduler
        if self.is_amp:
            self.amp_scaler[name] = GradScaler()
        model = self.model_to_device(self.student)
        self.models[name] = model

     

    def build_feature_connector(self, t_channel, s_channel):
        modules = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(t_channel),
             nn.ReLU(inplace=True)]

        for module in modules:
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
        return nn.Sequential(*modules)

    def calc_cd_loss(self):
        t_feats = self.teacher_w.get_mid_features()
        s_feats = self.student.get_mid_features()
        losses = []
        for connector, s_feat, target in zip(self.connectors, s_feats, t_feats):
            s_act = connector(s_feat)
            source, target = s_act, target.detach()
            source = source.mean(dim=(2, 3), keepdim=False)
            target = target.mean(dim=(2, 3), keepdim=False)
            loss = torch.mean(torch.pow(source - target, 2))
            losses.append(loss)
        return sum(losses)

    def cal_student_loss(self, fake_s, fake_t_w, fake_t_d):
        fake_ts = [fake_t_d.detach(), fake_t_w.detach()]
        lambda_ssim = 10
        lambda_style = 10
        loss_g_student = 0
        for fake_t in fake_ts:
            ssim_loss = SSIMNet()
            loss_g_ssim = (1 - ssim_loss(fake_s, fake_t)) * lambda_ssim
            t_feats = self.vgg(fake_t)
            s_feats = self.vgg(fake_s)
            t_gram = [self._gram_mat(fmap) for fmap in t_feats]
            s_gram = [self._gram_mat(fmap) for fmap in s_feats]

            loss_g_style = 0
            for i in range(len(t_gram)):
                loss_g_style += lambda_style * F.l1_loss(s_gram[i], t_gram[i])
            s_recon, t_recon = s_feats[1], t_feats[1]
            lambda_feature = 10
            loss_g_feature = lambda_feature * F.l1_loss(s_recon, t_recon)
            # diff_i = torch.sum(torch.abs(self.Sfake_B[:, :, :, 1:] - self.Sfake_B[:, :, :, :-1]))
            # diff_j = torch.sum(torch.abs(self.Sfake_B[:, :, 1:, :] - self.Sfake_B[:, :, :-1, :]))
            # self.loss_G_tv = self.lambda_tv * (diff_i + diff_j)
            loss_g_student += loss_g_ssim + loss_g_style + loss_g_feature # + self.loss_G_tv
        
        lambda_cd = 10
        loss_g_student += self.calc_cd_loss() * lambda_cd
        return loss_g_student