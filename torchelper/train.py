import os
import random
from torchelper.utils.dist_util import get_rank
from torchelper.models.model_group import ModelGroup
from tqdm import tqdm
import torch 
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torchelper.utils.cls_utils import get_cls
from torchelper.data.base_dataset import get_data_loader
import torch.backends.cudnn as cudnn
import subprocess

def check_close_port(port):
    result = subprocess.run(['lsof', '-i:'+str(port)], stdout=subprocess.PIPE)
    out = result.stdout.decode('utf-8')

    lines = out.split('\n')
    if len(lines)<=1:
        return
    print(out)
    lines = lines[1:]
    for line in lines:
        arr = [s for s in line.split(' ') if len(s)>0]
        if len(arr)<2:
            continue
        pid = int(arr[1])
        os.system('kill '+str(pid))
        print("kill", pid)

def check_close_gpu_ids(gpu_ids):
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]
    print('kill:', gpu_ids)
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    out = result.stdout.decode('utf-8')

    lines = out.split('\n')
    if len(lines)<=1:
        return

    start_flag = False
    for line in lines:
        if not start_flag:
            if line.startswith('|=') and not '+' in line:
                start_flag = True
        else:
            line = ' '.join(line.split())
            arr = line.split(' ')
            if len(arr)<8:
                continue
            print(line)
            gpu_id = int(arr[1])
            pid = int(arr[4])
            for gid in gpu_ids:
                if gpu_id == gid:
                    os.system('kill '+str(pid))

def get_port(def_port):
    result = subprocess.run(['lsof', '-i:'+str(def_port)], stdout=subprocess.PIPE)
    out = result.stdout.decode('utf-8')

    lines = out.split('\n')
    if len(lines)<=1:
        return def_port
    return get_port(def_port+1)


def validate(net:ModelGroup, epoch:int, val_dataset):
    '''执行验证集
    :param net: ModelGroup子类实例
    :param epoch: int, 当前epoch
    :param val_dataset: 验证集
    :return: 返回描述
    :raises ValueError: 描述抛出异常场景
    '''
    if get_rank()==0:
        net.before_validate(epoch)
        # time.sleep(10)
        
        val_data:dict = {}
        
        for data in tqdm(val_dataset):
            res = net.validate(epoch, data)
            if res is None:
                continue
            for key, val in res.items():
                val_data[key] = val + val_data.get(key, 0)
        
        line = 'epoch: '+str(epoch)+', '
        if val_data is not None:
            val_arr = []
            for key, val in val_data.items():
                val_arr.append(key + ":" + str(val_data[key] * 1.0 / len(val_dataset)))
            line = line+', '.join(val_arr)
        val_data = net.after_validate(epoch)
        if val_data is not None:
            val_arr = []
            for key, val in val_data.items():
                val_arr.append(key + ":" + str(val_data[key]))
            line = line +','+ ', '.join(val_arr)
        print(line)
    torch.distributed.barrier()


def train(gpu_id, cfg, is_dist):
    is_amp = cfg.get('amp', False)
    train_dataset_cls = get_cls(cfg['train_dataset_cls'])
    val_dataset_cls = get_cls(cfg['val_dataset_cls'])
    train_dataset = train_dataset_cls(gpu_id, cfg)
    train_dataloader = get_data_loader(cfg['batch_size'], train_dataset, dist=is_dist)
    val_dataloader = get_data_loader(cfg['batch_size'], val_dataset_cls(gpu_id, cfg), dist=False)
    net:ModelGroup = get_cls(cfg['model_group_cls'])(cfg, gpu_id, True, is_dist, is_amp)
    net.set_dataset(train_dataset)
    dataset_size = len(train_dataloader)
    print('#training images = %d' % dataset_size)
    
    net.save_model(-1)
    for epoch in range(cfg['start_epoch'], cfg['total_epoch']):
        # validate(net, epoch, val_dataloader)
        net.set_train()
        if is_dist and gpu_id==0:
            enum_data = enumerate(tqdm(train_dataloader))
        else:
            enum_data =  enumerate(train_dataloader)
        for i, data in enum_data:
            net.forward_wrapper(epoch, i, data)
            # 计算loss
            net.criterion_wrapper()
            if is_dist:   # 多卡同步
                torch.distributed.barrier()
            net.backward_wrapper()
        net.update_learning_rate(epoch)
        net.save_model(epoch)
        validate(net, epoch, val_dataloader)


def train_worker(gpu_id, nprocs, cfg, is_dist, port):
    '''独立进程运行
    '''
    os.environ['NCCL_BLOCKING_WAIT']="1"
    os.environ['NCCL_ASYNC_ERROR_HANDLING']='1'
    random.seed(0)
    torch.manual_seed(0)
    cudnn.deterministic = True
    # 提升速度，主要对input shape是固定时有效，如果是动态的，耗时反而慢
    torch.backends.cudnn.benchmark = True
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:'+str(port),
                            world_size=len(cfg['gpu_ids']),
                            rank=gpu_id)

    torch.cuda.set_device(gpu_id)
    # 按batch分割给各个GPU
    cfg['batch_size'] = int(cfg['batch_size'] / nprocs)
    train(gpu_id, cfg, is_dist)

def train_main(cfg):
    check_close_gpu_ids(cfg['ori_gpu_ids'])
    # check_close_port(cfg['port'])
    gpu_nums = len(cfg['gpu_ids'])
    # if gpu_nums>1:
    port = get_port(cfg['port'])
    print('init port:', port)
    mp.spawn(train_worker, nprocs=gpu_nums, args=(gpu_nums, cfg, True, port))
    # else:
    #     train(cfg['gpu_ids'][0], cfg, False)
