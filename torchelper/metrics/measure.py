
import cv2
import numpy as np
import torch

class Metric():
    def __init__(self):
        pass

    def cal(self, inp, tar):
        pass
    
  

    def img_to_batch_tensor(self, data, device):
        if isinstance(data, np.ndarray):
            if len(data.shape)==4: # 如果是4维，则认为是[b, h, w, c]
                data = np.transpose(data, (0, 3, 1, 2)).astype(np.float32)
                return  torch.from_numpy(data).to(device)
            elif len(data.shape)==3: # 如果是3维，则认为是[h, w, c]
                data = np.expand_dims(data, 0)
                data = np.transpose(data, (0, 3, 1, 2)).astype(np.float32)
                return  torch.from_numpy(data).to(device)
        raise Exception("Unknow type "+str(type(data)))
    
    def align_size(self, uint8_a, uint8_b):
        height, width = uint8_a.shape[:2]
        uint8_b = cv2.resize(uint8_b, (width, height))
        return uint8_a, uint8_b

