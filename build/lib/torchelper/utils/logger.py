
from .dist_util import master_only

@master_only
def debug(*msg):
    print(msg)

@master_only
def log(*msg):
    print(msg)

@master_only
def warn(*msg):
    print(msg)