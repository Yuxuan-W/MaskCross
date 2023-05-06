import os
import time
import random
import torch
import MinkowskiEngine as ME
from tqdm import tqdm

from datasets.semseg import ScannetDataset
from datasets.utils import VoxelizeCollate
from models.mask3d import Mask3D
from util.logger import setup_logger
from util.inference import instance_inference, semantic_inference
from util.evaluate import evaluate_instance, log_instance_results, evaluate_semantic, log_semantic_results


def worker_init_fn(worker_id):
    random.seed(time.time() + worker_id)


def main_tester(argss):
    global args
    args = argss

    for k, v in args.test_specialized.items():
        args[k] = v

    global logger
    logger = setup_logger("test", args.save_path, "test.txt")
    logger.info(args)

    model = Mask3D(args).cuda()

    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        # new_dict = {}
        # for k, v in checkpoint['state_dict'].items():
        #     if k[:5] == 'model' and k.split('.')[2] != 'final':
        #         new_dict[k[6:]] = v
        # model.load_state_dict(new_dict)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    test_data = ScannetDataset(label_db_filepath=args.label_db_filepath, color_mean_std=args.color_mean_std,
                               mode='validation', add_colors=True, add_normals=False, add_instance=True,
                               point_per_cut=100, add_raw_coordinates=True, num_labels=args.num_labels,
                               ignore_label=args.ignore_label, filter_out_classes=args.filter_out_classes,
                               label_offset=len(args.filter_out_classes))
    test_collator = VoxelizeCollate(mode='validation', ignore_label=args.ignore_label,
                                    num_queries=args.num_object_queries, voxel_size=args.voxelSize,
                                    filter_out_classes=args.filter_out_classes,
                                    label_offset=len(args.filter_out_classes), task=args.task)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                             drop_last=False, collate_fn=test_collator,
                                             sampler=None)
    model.eval()
    with torch.no_grad():
        preds = {}
        pbar = tqdm(total=len(test_loader))
        for i, (data, target, file_names) in enumerate(test_loader):
            inverse_maps = data.inverse_maps
            target_full = data.target_full
            original_colors = data.original_colors
            data_idx = data.idx
            original_normals = data.original_normals
            original_coordinates = data.original_coordinates
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]
            data = ME.SparseTensor(coordinates=data.coordinates, features=data.features, device='cuda')

            for tgt in target:
                for k, v in tgt.items():
                    tgt[k] = v.cuda(non_blocking=True)

            output = model.forward(data, point2segment=[target[i]['point2segment'] for i in range(len(target))],
                                   raw_coordinates=raw_coordinates, is_eval=True)

            if args.save_visualizations:
                backbone_features = output['backbone_features'].F.detach().cpu().numpy()
                from sklearn import decomposition
                pca = decomposition.PCA(n_components=3)
                pca.fit(backbone_features)
                pca_features = pca.transform(backbone_features)
                rescaled_pca = 255 * (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())

            if args.task == "instance_segmentation":
                preds.update(instance_inference(args, output, target, target_full, inverse_maps, file_names,
                                                original_coordinates, original_colors, original_normals,
                                                raw_coordinates, data_idx,
                                                backbone_features=rescaled_pca if args.save_visualizations else None,
                                                test_mode='validation',
                                                remap_model_output=test_data.remap_model_output))
            elif args.task == "semantic_segmentation":
                preds.update(semantic_inference(args, output, target, target_full, inverse_maps, file_names))
            else:
                raise ValueError

            pbar.update(1)

        if args.task == "instance_segmentation":
            ap_3d = evaluate_instance(args, preds)
            log_instance_results(ap_3d, logger)
        elif args.task == "semantic_segmentation":
            iou_3d = evaluate_semantic(args, preds)
            log_semantic_results(iou_3d, logger)
        else:
            raise ValueError