
def get_cls(class_str):
    '''动态创建对象
    :param class_str: str, 类名称
    '''
    arr = class_str.split('.')
    g_module = __import__('.'.join(arr[0:-1]) , fromlist = True)
    g_cls = getattr(g_module, arr[-1])
    return g_cls

def new_cls(class_str, *arg):
    '''动态创建对象
    :param class_str: str, 类名称
    :return: Model, class_str对应的模型类
    '''
    g_cls = get_cls(class_str)
    return g_cls(*arg)

def auto_new_cls(cfg:dict):
    """自动创建类，从cfg中读取cls类，并将cfg作为参数
    """
    return new_cls(cfg['cls'], *[cfg])