

# from skimage.metrics import peak_signal_noise_ratio as compare_ssim  
from .measure import Metric
import numpy as np
import cv2
import torch.nn as nn
import torch
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(window_2d.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # C1 = 0.01 ** 2
    # C2 = 0.03 ** 2

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)





def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)





class SSIM(Metric):
    def __init__(self):
        super().__init__()

    def cal(self, inp, tar):
        if len(inp)==3:
            inp, tar = np.array([inp]), np.array([tar])
        
        inp = inp.transpose((0, 3, 1, 2)).astype(np.float64)
        tar = tar.transpose((0, 3, 1, 2)).astype(np.float64)
        inp = torch.from_numpy(inp)
        tar = torch.from_numpy(tar)
        res = ssim(inp, tar)
        res = res.detach().cpu().numpy()
        return res




class SSIMNet(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

 
# class SSIM(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def _ssim(self, img1, img2):
#         """Calculate SSIM (structural similarity) for one channel images.
#         It is called by func:`calculate_ssim`.
#         Args:
#             img1 (ndarray): Images with range [0, 255] with order 'HWC'.
#             img2 (ndarray): Images with range [0, 255] with order 'HWC'.
#         Returns:
#             float: ssim result.
#         """

#         C1 = (0.01 * 255)**2
#         C2 = (0.03 * 255)**2

#         img1 = img1.astype(np.float64)
#         img2 = img2.astype(np.float64)
#         kernel = cv2.getGaussianKernel(11, 1.5)
#         window = np.outer(kernel, kernel.transpose())

#         mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
#         mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#         mu1_sq = mu1**2
#         mu2_sq = mu2**2
#         mu1_mu2 = mu1 * mu2
#         sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#         sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#         sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#         ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#         return ssim_map.mean()

    
#     def calculate_ssim(self, img1, img2):
#         """Calculate SSIM (structural similarity).
#         Ref:
#         Image quality assessment: From error visibility to structural similarity
#         The results are the same as that of the official released MATLAB code in
#         https://ece.uwaterloo.ca/~z70wang/research/ssim/.
#         For three-channel images, SSIM is calculated for each channel and then
#         averaged.
#         Args:
#             img1 (ndarray): Images with range [0, 255].
#             img2 (ndarray): Images with range [0, 255].
#         Returns:
#             float: ssim result.
#         """
#         img1 = img1.astype(np.float64)
#         img2 = img2.astype(np.float64)
#         ssims = []
#         for i in range(img1.shape[2]):
#             ssims.append(self._ssim(img1[..., i], img2[..., i]))
#         return np.array(ssims).mean()

#     # def ssim(self, img_a, img_b):
#         # '''测量ssim
#         # :param img_a: np.array, dtype=np.uint8, 输入图
#         # :param img_b: np.array, dtype=np.uint8, GT
#         # :return: 返回ssim值
#         # '''
#         # # multichannel: If True, treat the last dimension of the array as channels.
#         # # Similarity calculations are done independently for each channel then averaged.
#         # print(img_a.shape)
#         # print(img_b.shape)
#         # score, __ = compare_ssim(img_a, img_b, full=True, multichannel=True)
#         # return score

#     def cal(self, inp, tar):
#         if len(inp)==3:
#             batch_size = 1
#             inp, tar = [inp], [tar]
#         else:
#             batch_size = inp.shape[0]
#         sum_v = 0
#         for i in range(batch_size):
#             sum_v += self.calculate_ssim(inp[i], tar[i])
#         res = sum_v * 1.0 / batch_size
#         print(res)
#         return res