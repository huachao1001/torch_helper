
from .dist_util import master_only

@master_only
def debug(*msg):
    print(msg)