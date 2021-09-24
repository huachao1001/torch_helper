import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
def get_data_loader(batch_size:int, dataset:Dataset, num_workers:int=0, pin_memory=True, dist:bool=True):
    '''封装DataLoader
    :param batch_size: int
    :param dataset: Dataset
    :param num_workers: int
    :param dist: bool, 是否是分布式采集数据
    '''
    sampler = None
    if dist:
        sampler = DistributedSampler(dataset) # 这个sampler会自动分配数据到各个gpu上
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        drop_last=True,   # 多余的部分去除
    )
    return dataloader

class BaseDataset(Dataset):
    def imread(self, path, is_rgb:bool=True):
        '''读取图片数据
        :param is_rgb: bool, 返回格式是否是rgb
        :return: np.ndarray, dtype=np.uint8,返回图片数据
        '''
        img = cv2.imread(path)
        if img is None:
            return None
        #取前3个通道
        img = img[:, :, 0:3]
        if is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def imwrite(self, img, path, is_rgb:bool=True):
        if is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, img)

    def norm(self, uint8_img:np.ndarray, low:float=-1.0, high:float=1.0):
        '''归一化
        :param uint8_img: dtype=np.uint8, 输入图像数据
        :param low: 归一化后的最小值
        :param high: 归一化后的最大值
        :return: 归一化后的数据，数据类型为float32
        '''
        assert high > low, "high must be greater than low"
        data = uint8_img.astype(np.float32)
        data = data / 255.0 * (high - low) + low
        return data.astype(np.float32)

    def unnorm(self, float_img:np.ndarray, low:float=-1, high:float=1.0):
        '''逆归一化
        :param uint8_img: dtype=np.uint8, 输入图像数据
        :param low: 归一化后的最小值
        :param high: 归一化后的最大值
        :return: 逆归一化后的数据，数据类型为uint8
        '''
        assert high > low, "high must be greater than low"
        data = float_img /  (high - low) * 255
        data = np.clip(data, 0, 255).astype(np.uint8)
        return data

    def to_tensor(self, im_hwc):
        '''将numpy.ndarray的hwc排列的图片转为tensor
        :param im_hwc: numpy.ndarray的hwc排列的图片
        :return: tenorsor
        '''
        im_hwc = im_hwc.astype(np.float32)
        img = im_hwc.transpose((2, 0, 1))
        img = torch.from_numpy(img)
        return img

    def is_img_name(self, name):
        '''判断名称是否是图片类型
        :param name: str, 名称
        :return: bool, 判断结果
        '''
        lower = name.lower()
        return lower.endswith('.png') or \
               lower.endswith('.jpg') or \
               lower.endswith('.jpeg')