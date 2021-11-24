import os
import torch
import time
import numpy as np
from .callback import Callback
from torchelper.models.base_model import BaseModel
from torchelper.utils.dist_util import master_only, get_bare_model

class ReduceLROnPlateau(Callback):
    """当指定指标没有提升时，自动下降学习率。
    patience个epoch后，如果指标没提升，将学习率调整为`CurLR = CurLR * factor`.

    示例:
    ```python
    class ExampleModel(BaseModel):
        ...
        def get_metric_dict(self):
            return {'ex_loss', self.ex_loss}
        ...

    model = ExampleModel(...)
    model.add_cb(ReduceLROnPlateau(monitor='ex_loss', factor=0.2,
                                    patience=5, min_lr=0.001))
    ```
    Args:
        monitor: str, 监控的指标名称，必须是对应Model的get_metric函数返回字典中包括的key
        factor: float, 学习率下降倍数，`new_lr = lr * factor`.
        patience: int, 指定指标没有提升的epoch次数，超过此次数学习率将会下降。
        mode: str, 模式，只能取`{'min', 'max'}`中一个. 在`'min'` 模式下, monitor的
            值越低代表指标性能越好，在`'max'`模式下，monitor值越大代表性能指标越好。
        min_delta: int, 表示阈值， 如果两个epoch对应的指标值处于这个阈值范围之内，表示指标没有提升。
        cooldown: int, cooldown个epoch为冷却期， 冷却期间不进行性能指标对比和学习率下降。
        min_lr: lr的最小衰减。 如果新旧lr之间的差异小于eps，则忽略更新。 默认值：1e-8。
    """
    def __init__(self,
               monitor,
               factor=0.1,
               patience=10,
               mode='min',
               min_delta=1e-4,
               cooldown=0,
               min_lr=0):
        
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()
    
    def _reset(self):
        """重置等待次数和冷却次数
        """
        if self.mode not in ['min', 'max']:
            print('Learning rate reduction mode %s is unknown')
            exit()
        if (self.mode == 'min'):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def get_lr(self, model:BaseModel):
        optimizer = model.get_optimizer()
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                return param_group['lr']

    def set_lr(self, model:BaseModel, lr:float):
        optimizer = model.get_optimizer()
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def on_begin_train(self, model:BaseModel):
        self._reset()
    
    def on_end_epoch(self, model:BaseModel, epoch:int):
        metric:dict = model.get_metric_dict()
        current = metric.get(self.monitor, None)
        if current is None:
            print('Learning rate reduction is conditioned on metric `%s` '
                        'which is not available. Available metrics are: %s',
                        self.monitor, ','.join(list(metric.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = self.get_lr(model)
                    if old_lr is None:
                        print('can not get current learn rate !!!')
                        exit()
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.set_lr(model, new_lr)
                        print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                    'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0
