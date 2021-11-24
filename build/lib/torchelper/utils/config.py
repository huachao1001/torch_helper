'''读取配置文件
'''
import os
import yaml
import sys
def load_default():
    return {
        'rank': 0,
        'seed': 0,
        'port': 22021,
        'amp' : True,
        'gpu_ids': '0'
    }

def replace_home_dir(cfg):
    home = os.path.expanduser("~")
    if isinstance(cfg, str) and cfg.startswith('~/'):
        return os.path.join(home, cfg[2:])
    elif isinstance(cfg, list) or isinstance(cfg, tuple):
        new_v = []
        for v in cfg:
            v = replace_home_dir(v)
            new_v.append(v)
        if isinstance(cfg, tuple):
            new_v = tuple(new_v)
        return new_v
    elif isinstance(cfg, dict): 
        for key, val in cfg.items():
            val = replace_home_dir(val)
            cfg[key] = val
        return cfg 
    return cfg
    


def load_cfg(path):
    '''加载配置参数
    '''
    file_path = os.path.abspath(__file__)
    file_dir = os.path.dirname(file_path)
    # with open(os.path.join(file_dir, '../default.yml'), 'r', encoding='utf-8') as file:
    #     base_cfg = yaml.safe_load(file)
    base_cfg = load_default()
    with open(path, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    base_cfg.update(cfg)
    base_cfg = replace_home_dir(base_cfg)
    return base_cfg


def parse_args():
    '''解析命令行参数
    '''
    if len(sys.argv) < 2:
        print('Please pass in the yml file, the example is as follows:')
        print('  python train.py your_cfg.yml')
        exit()
    path = sys.argv[1]
    return load_cfg(path)

def init_cfg():
    '''读取并初始化配置文件
    :return: dict
    '''
    cfg = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_ids']
    cfg['ori_gpu_ids'] = [int(gpu_id.strip()) for gpu_id in cfg['gpu_ids'].split(",")]
    cfg['gpu_ids'] = [i for i in range(len(cfg['gpu_ids'].split(",")))]
    # check path
    if not os.path.exists( cfg['ckpt_dir']+"/train_out"):
        os.makedirs(cfg['ckpt_dir']+"/train_out")
    return cfg

def merge_cfg(cfg, default_cfg):
    tmp = default_cfg.copy()
    tmp.update(cfg)
    return tmp