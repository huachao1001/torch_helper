
import os
import cv2
from abc import ABCMeta, abstractmethod
from .base_dataset import BaseDataset

class DirDataset(BaseDataset):
    '''单目录数据读取
    '''
    def __init__(self, dir_path, dst_wh=None, img_dir=None, label_name=None, cfg=None):
        self.data_root = dir_path
        self.cfg = cfg
        self.img_dir = img_dir
        self.label_name = label_name
        self.dst_wh = dst_wh
        self.labels = {}
        self.load_data()
        self.size = len(self.names)

    def load_data(self):
        #加载label
        if self.label_name is not None:
            self.labels = self.parse_label(os.path.join(self.data_root , self.label_name))
        if self.img_dir is not None:
            self.data_root = os.path.join(self.data_root, self.img_dir)
        self.names = sorted(os.listdir(self.data_root))

    def process_img(self, name, img, label):
        return name, img, label

    def __getitem__(self, index):
        name = self.names[index]
        label = self.labels.get(name, None)
        img_path = os.path.join(self.data_root,  name)
        img = self.imread(img_path)
        if self.dst_wh is not None:
            if isinstance(self.dst_wh, list):
                img = cv2.resize(img, self.dst_wh)
            else:
                img = cv2.resize(img, (self.dst_wh, self.dst_wh))
        return self.process_img(name, img, label)

    def __len__(self):
        return self.size


class MultiDirDataset(BaseDataset, metaclass=ABCMeta):
    '''多目录读取
    '''

    def __init__(self, root_dir_path, sub_dirs=None, img_dir=None, label_name=None, dst_wh=None, cfg=None):
        self.data_root = root_dir_path
        self.img_dir = img_dir
        self.sub_dirs = sub_dirs
        self.label_name = label_name
        self.cfg = cfg
        self.dst_wh = dst_wh
        self.labels = {}
        self.names = []
        self.load_data()
        self.size = len(self.names)

    def load_data(self):
        if self.sub_dirs is None:
            dirs = os.listdir(self.data_root)
        else:
            dirs = self.sub_dirs
        for dir in dirs:
            dir_path = os.path.join(self.data_root, dir)
            if not os.path.isdir(dir_path):
                continue
            pre_name, img_dir = dir, dir_path
            #判断目录下是否有图片子目录
            if self.img_dir is not None:
                img_dir = os.path.join(img_dir, self.img_dir)
                pre_name = pre_name + '/' + self.img_dir
            #获取图片子目录+名称
            for name in os.listdir(img_dir):
                if self.is_img_name(name):
                    self.names.append(pre_name + '/' + name)
            #加载label
            if self.label_name is not None:
                self.labels = self.parse_label(self.labels, dir, os.path.join(dir_path, self.label_name))


    @abstractmethod
    def parse_label(self, label_dict, dir_name, label_path):
        '''解析label
        :param label_dict: 待更新的label字典，注意：key为目录+图片名称，例如：dir_name/imgs/img1.png，
                因为根目录是通用的，为了节约空间，省去根目录。
        :param dir_name: 目录名称
        :param label_path: label路径
        :return: dict, 返回更新后的字典
        '''
        return label_dict

    def process_img(self, name, img, label):
        return name, img, label

    def __getitem__(self, index):

        name = self.names[index]
        label = self.labels.get(name, None)
        img_path = os.path.join(self.data_root,  name)
        # img is RGB
        img = self.imread(img_path)
        if self.dst_wh is not None:
            if isinstance(self.dst_wh, list):
                img = cv2.resize(img, self.dst_wh)
            else:
                img = cv2.resize(img, (self.dst_wh, self.dst_wh))
        return self.process_img(name, img, label)

    def __len__(self):
        return self.size
