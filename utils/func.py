import time
from contextlib import contextmanager
import multiprocessing as mp
import random
import os
import numpy as np


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def chunk_list(big_list, chunks=2):
    for i in range(0,len(big_list),chunks):
        yield big_list[i:i+chunks]


def multi_process(compute,iter_list,process=4):
    """
    定义进程池，实现多进程并行。
    param:
        compute: 计算模块，传入函数。
        iter_list: 不同计算模块对应的输入数据。
        process: 进程个数，不超过CPU逻辑核心数。
    """
    results = []
    with mp.Pool(processes=process) as pool:
        workers = pool.imap_unordered(compute, iter_list)
        for result in workers:
            results.append(result)
    return results


def setRandomSeed(seed=2022): 
    random.seed(seed) 
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed) 
    # torch.manual_seed(seed) 
    # if torch.cuda.is_available(): 
    # torch.cuda.manual_seed(seed) 
    # torch.cuda.manual_seed_all(seed) 
    # torch.backends.cudnn.deterministic = True 
    # torch.backends.cudnn.benchmark = False 
    # torch.backends.cudnn.enabled = True 