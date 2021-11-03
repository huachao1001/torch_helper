
from torchelper.models.base_model import BaseModel

class Callback():
    def __init__(self):
        pass

    def on_begin_train(self, model:BaseModel):
        pass

    def on_end_train(self, model:BaseModel):
        pass

    def on_begin_epoch(self, model:BaseModel, epoch:int):
        pass

    def on_end_epoch(self, model:BaseModel, epoch:int):
        pass

    def on_begin_step(self, model:BaseModel, epoch:int, step:int):
        pass

    def on_end_step(self, model:BaseModel, epoch:int, step:int):
        pass