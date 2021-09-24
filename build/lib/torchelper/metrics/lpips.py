
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torch
from torchelper.utils.dist_util import get_rank
from .measure import Metric
import lpips


class LPIPS(Metric):
    def __init__(self, gpu_id, net='alex'):
        '''
        :param net: str, 网络结构
        '''
        super().__init__()
        self.device = torch.device('cuda:'+str(gpu_id) if gpu_id is not None else 'cpu')
        self.model = lpips.LPIPS(net=net)
        self.model = self.model.to(self.device)
    
 
    def cal(self, inp, tar):
        '''测量lpips
        :param inp: np.array, dtype=np.uint8, 输入图
        :param tar: np.array, dtype=np.uint8, GT
        :return: 返回lpips值
        '''
        inp = self.img_to_batch_tensor(inp, self.device)
        tar = self.img_to_batch_tensor(tar, self.device)
        inp = inp / 127.5 - 1
        tar = tar / 127.5 - 1
        
        dist01 = self.model.forward(inp, tar).detach().cpu().numpy()
        val = np.mean(dist01)
        return val