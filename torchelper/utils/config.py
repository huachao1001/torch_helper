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
    #reset home
    home = os.path.expanduser("~")
    for key, val in base_cfg.items():
        if isinstance(val, str) and val.startswith('~/'):
            base_cfg[key] = os.path.join(home, val[2:])
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
