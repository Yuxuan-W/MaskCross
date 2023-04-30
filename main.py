import os
import time
import random
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
from util import config
from tool.trainer import main_trainer
from tool.tester import main_tester


def worker_init_fn(worker_id):
    random.seed(time.time() + worker_id)


def get_parser():
    parser = argparse.ArgumentParser(description='BPNet')
    parser.add_argument('--config', type=str, default='config/scannet3d_5cm.yaml', help='config file')
    parser.add_argument('opts', help='see config/scannet/bpnet_5cm.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    localtime = time.localtime()
    args.save_path += "_{mon}-{day}_{hour}-{min}".format(mon=localtime.tm_mon,
                                                               day=localtime.tm_mday,
                                                               hour=localtime.tm_hour,
                                                               min=localtime.tm_min)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(os.path.join(args.save_path, 'model')):
        os.mkdir(os.path.join(args.save_path, 'model'))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)

    if len(args.train_gpu) == 1:
        args.sync_bn_2d = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if 'train' in args.mode:
        if args.multiprocessing_distributed:
            args.world_size = args.ngpus_per_node * args.world_size
            mp.spawn(main_trainer, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
        else:
            main_trainer(args.train_gpu[0], args.ngpus_per_node, args)
        args.model_path = os.path.join(args.save_path, 'model/model_best.pth.tar')

    if 'test' in args.mode:
        main_tester(args)


if __name__ == '__main__':
    main()
