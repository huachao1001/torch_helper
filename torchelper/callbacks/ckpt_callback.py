import os
import torch
import time
 
from .callback import Callback
from torchelper.models.base_model import BaseModel
from torchelper.utils.dist_util import master_only, get_bare_model

class CkptCallback(Callback):
    def __init__(self, ckpt_dir, name, restore_epoch=-1, max_ckpts=10, save_per_secs=2*60*60, strict=False, save_before_train=True):
        super().__init__()
        self.name = name
        self.strict = strict
        self.ckpt_dir = ckpt_dir
        self.restore_epoch = restore_epoch
        self.epochs = []
        self.last_save_time = -1
        self.max_ckpts=max_ckpts
        self.save_per_secs = save_per_secs
        self.save_before_train = save_before_train
    
    def load_weights(self, model, epoch):
        # 1. load weights
        save_filename = "%s_weights_%s.pth" % (epoch, self.name)
        save_path = os.path.join(self.ckpt_dir, save_filename)
        if os.path.exists(save_path):
            weights = torch.load(save_path, map_location=lambda storage, loc: storage)
            weights = get_bare_model(model).remap_weights_name(weights)
            get_bare_model(model).load_state_dict(weights, strict=self.strict)
            print("success load model:"+ save_path)
        else:
            print("%s not exists yet!" % save_path)
            return False
        
        # 2. load optimizer
        optimizer = get_bare_model(model).get_optimizer()
        if optimizer is not None:
            save_filename = "%s_optimizer_%s.pth" % (epoch, self.name)
            save_path = os.path.join(self.ckpt_dir, save_filename)
            if not os.path.isfile(save_path):
                print("%s not exists yet!" % save_path)
                return False
            else:
                weights = torch.load(save_path, map_location=lambda storage, loc: storage)
                try:
                    optimizer.load_state_dict(weights)
                except:
                    print('Failed to load optimizer parameters')
                    return False
                print("success load optimizer:", save_path)
        return True

    @master_only
    def save_model(self, model:BaseModel, epoch):
        #1. save optimizer
        optimizer = model.get_optimizer()
        if optimizer is not None:
            save_opt_name = "%s_optimizer_%s.pth" % (epoch, self.name)
            save_opt_path = os.path.join(self.ckpt_dir, save_opt_name)
            torch.save(optimizer.state_dict(), save_opt_path)
            print('save:', save_opt_path)

        #2. save weights
        save_weight_filename = "%s_weights_%s.pth" % (epoch, self.name)
        save_weight_path = os.path.join(self.ckpt_dir, save_weight_filename)
        torch.save(get_bare_model(model).state_dict(), save_weight_path)
        print('save:', save_weight_path)

        #3. clear old
        if self.max_ckpts<=0: #不清除
            return
        # 每间隔max_time时间保存一次
        if time.time() - self.last_save_time > self.save_per_secs:
            self.last_save_time = time.time()
        else:
            if epoch not in self.epochs:
                self.epochs.append(epoch)
            while len(self.epochs) > self.max_ckpts:
                opt_name = '%s_optimizer_%s.pth'%(self.epochs[0], self.name)
                weight_name = '%s_weights_%s.pth'%(self.epochs[0], self.name)
                opt_path = os.path.join(self.ckpt_dir, opt_name)
                weight_path = os.path.join(self.ckpt_dir, weight_name)
                print(opt_path)
                if os.path.exists(opt_path):
                    os.remove(opt_path)
                if os.path.exists(weight_path):
                    os.remove(weight_path)
                del self.epochs[0]


    def on_begin_train(self, model:BaseModel):
        self.load_weights(model, self.restore_epoch)
        if self.save_before_train:
            self.save_model(model, -1)
        


    def on_end_train(self, model:BaseModel):
        pass

    def on_begin_epoch(self, model:BaseModel, epoch:int):
        pass

    def on_end_epoch(self, model:BaseModel, epoch:int):
        self.save_model(model, epoch)

    def on_begin_step(self, model:BaseModel, epoch:int, step:int):
        pass

    def on_end_step(self, model:BaseModel, epoch:int, step:int):
        pass