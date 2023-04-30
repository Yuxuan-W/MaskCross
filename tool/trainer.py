import os
import time
import random
import torch
import logging
import wandb
import torch.distributed as dist
import MinkowskiEngine as ME
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from models.mask3d import Mask3D
from models.matcher import HungarianMatcher
from models.criterion import SetCriterion
from datasets.semseg import ScannetDataset
from datasets.utils import VoxelizeCollate
from util.logger import setup_logger
from util.misc import save_checkpoint, AverageMeter
from util.inference import eval_instance_step
from util.evaluate import evaluate_instance, log_instance_results


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(time.time() + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main_trainer(gpu, ngpus_per_node, argss):
    global args
    args = argss

    if main_process():
        global logger
        logger = setup_logger("train", args.save_path, "train.txt")
        logger.info(args)

        if 'debug' not in args.save_path:
            wandb.init(project='MaskCross', config=args)
            wandb.run.name = args.save_path
            wandb.run.save()

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model = Mask3D(args)

    if args.weight:
        assert os.path.isfile(args.weight)
        if main_process():
            logger.info("=> loading weight '{}'".format(args.weight))
        checkpoint = torch.load(args.weight)
        new_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if k[:5] == 'model' and k.split('.')[2] != 'final':
                new_dict[k[6:]] = v
        model.load_state_dict(new_dict)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])

    # if args.weight:
    #     assert os.path.isfile(args.weight)
    #     if main_process():
    #         logger.info("=> loading weight '{}'".format(args.weight))
    #     checkpoint = torch.load(args.weight)
    #     model.load_state_dict(checkpoint['state_dict'])

    train_data = ScannetDataset(label_db_filepath=args.label_db_filepath, color_mean_std=args.color_mean_std,
                                mode='train', add_colors=True, add_normals=False, add_instance=True, point_per_cut=0,
                                add_raw_coordinates=True, num_labels=args.num_labels, ignore_label=args.ignore_label,
                                filter_out_classes=args.filter_out_classes, label_offset=len(args.filter_out_classes),
                                volume_augmentations_path=args.volume_augmentations_path,
                                image_augmentations_path=args.image_augmentations_path)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
    train_collator = VoxelizeCollate(mode='train', ignore_label=args.ignore_label, num_queries=args.num_object_queries,
                                     voxel_size=args.voxelSize, filter_out_classes=args.filter_out_classes,
                                     label_offset=len(args.filter_out_classes))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True, worker_init_fn=worker_init_fn,
                                               collate_fn=train_collator)

    val_data = ScannetDataset(label_db_filepath=args.label_db_filepath, color_mean_std=args.color_mean_std,
                              mode='validation', add_colors=True, add_normals=False, add_instance=True, point_per_cut=100,
                              add_raw_coordinates=True, num_labels=args.num_labels, ignore_label=args.ignore_label,
                              filter_out_classes=args.filter_out_classes, label_offset=len(args.filter_out_classes))

    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if args.distributed else None
    val_sampler = None
    val_collator = VoxelizeCollate(mode='validation', ignore_label=args.ignore_label, num_queries=args.num_object_queries,
                                   voxel_size=args.voxelSize, filter_out_classes=args.filter_out_classes,
                                   label_offset=len(args.filter_out_classes))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                             shuffle=False, num_workers=args.workers, pin_memory=True,
                                             drop_last=False, collate_fn=val_collator,
                                             sampler=val_sampler)

    criterion = SetCriterion(args, HungarianMatcher(args.cost_class, args.cost_mask, args.cost_dice)).cuda()

    optimizer = AdamW(lr=args.base_lr, weight_decay=args.weight_decay, params=model.parameters())
    scheduler = OneCycleLR(optimizer, epochs=args.epochs, max_lr=args.base_lr, steps_per_epoch=len(train_loader))
    best_ap = 0

    for epoch in range(args.epochs):
        model.train()
        if main_process():
            loss_meter = AverageMeter()
            pbar = tqdm(total=len(train_loader))
        for i, (data, target, file_names) in enumerate(train_loader):
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]
            data = ME.SparseTensor(coordinates=data.coordinates,
                                   features=data.features,
                                   device=model.device)
            for tgt in target:
                for k, v in tgt.items():
                    tgt[k] = v.cuda(non_blocking=True)

            output = model(data, point2segment=[target[i]['point2segment'] for i in range(len(target))],
                           raw_coordinates=raw_coordinates)

            losses = criterion(output, target, mask_type="segment_mask" if args.on_segment else "masks")

            for k in list(losses.keys()):
                if k in criterion.weight_dict:
                    losses[k] *= criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if main_process():
                loss_meter.update(loss.item())
                pbar.update(1)

        if main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Loss {loss:.4f}'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                 loss=loss_meter.avg))
            loss_meter.reset()
            if 'debug' not in args.save_path:
                wandb.log({'epoch': epoch,
                           'loss_train_batch': loss.item()})
                for k in list(losses.keys()):
                    wandb.log({k: losses[k].item()})

        if ((epoch + 1) % args.eval_freq == 0) and args.evaluate:
            model.eval()
            with torch.no_grad():
                if main_process():
                    preds = {}
                    pbar = tqdm(total=len(val_loader))
                for i, (data, target, file_names) in enumerate(val_loader):
                    inverse_maps = data.inverse_maps
                    target_full = data.target_full
                    original_colors = data.original_colors
                    data_idx = data.idx
                    original_normals = data.original_normals
                    original_coordinates = data.original_coordinates
                    raw_coordinates = data.features[:, -3:]
                    data.features = data.features[:, :-3]
                    data = ME.SparseTensor(coordinates=data.coordinates, features=data.features, device=model.device)

                    for tgt in target:
                        for k, v in tgt.items():
                            tgt[k] = v.cuda(non_blocking=True)

                    output = model.forward(data, point2segment=[target[i]['point2segment'] for i in range(len(target))],
                                           raw_coordinates=raw_coordinates, is_eval=True)

                    losses = criterion(output, target, mask_type="segment_mask" if args.on_segment else "masks")

                    for k in list(losses.keys()):
                        if k in criterion.weight_dict:
                            losses[k] *= criterion.weight_dict[k]
                        else:
                            # remove this loss if not specified in `weight_dict`
                            losses.pop(k)
                    loss = sum(losses.values())

                    if main_process():
                        preds.update(eval_instance_step(args, output, target, target_full, inverse_maps, file_names,
                                                        original_coordinates, original_colors, original_normals,
                                                        raw_coordinates, data_idx,
                                                        backbone_features=None,
                                                        test_mode='validation',
                                                        remap_model_output=val_data.remap_model_output))
                        pbar.update(1)

                if main_process():
                    ap_3d = evaluate_instance(preds, os.path.join(args.data_root, 'instance_gt/validation'))
                    log_instance_results(ap_3d, logger)
                    if "debug" not in args.save_path:
                        wandb.log({'all_ap': ap_3d['all_ap'],
                                   'all_ap_50': ap_3d['all_ap_50%'],
                                   'all_ap_25': ap_3d['all_ap_25%']})

        if ((epoch + 1) % args.save_freq == 0) and main_process():
            is_best = ap_3d['all_ap_50%'] >= best_ap
            best_ap = max(best_ap, ap_3d['all_ap_50%'])
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best': best_ap
                }, is_best, os.path.join(args.save_path, 'model'), f'epoch{epoch + 1}.pth.tar'
            )