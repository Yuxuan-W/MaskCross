import torch
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from torch_scatter import scatter_mean
from util.visualization import save_visualizations


def get_full_res_mask(mask, inverse_map, point2segment_full, on_segments, is_heatmap=False):
    mask = mask.detach().cpu()[inverse_map]  # full res

    if on_segments and is_heatmap == False:
        mask = scatter_mean(mask, point2segment_full, dim=0)  # full res segments
        mask = (mask > 0.5).float()
        mask = mask.detach().cpu()[point2segment_full.cpu()]  # full res points

    return mask


def get_mask_and_scores(args, mask_cls, mask_pred, num_queries=100, num_classes=18, device=None):
    if device is None:
        device = mask_cls.device
    labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

    if args.topk_per_image != -1:
        scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(args.topk_per_image, sorted=True)
    else:
        scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(num_queries, sorted=True)

    labels_per_query = labels[topk_indices]
    topk_indices = topk_indices // num_classes
    mask_pred = mask_pred[:, topk_indices]

    result_pred_mask = (mask_pred > 0).float()
    heatmap = mask_pred.float().sigmoid()

    mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)
    score = scores_per_query * mask_scores_per_image
    classes = labels_per_query

    return score, result_pred_mask, classes, heatmap


def eval_instance_step(args, output, target_low_res, target_full_res, inverse_maps, file_names, full_res_coords,
                       original_colors, original_normals, raw_coords, idx, remap_model_output,
                       backbone_features=None, test_mode='validation'):
    preds = {}

    label_offset = len(args.filter_out_classes)
    prediction = output['aux_outputs']
    prediction.append({
        'pred_logits': output['pred_logits'],
        'pred_masks': output['pred_masks']
    })

    prediction[-1]['pred_logits'] = torch.functional.F.softmax(
        prediction[-1]['pred_logits'],
        dim=-1)[..., :-1]  # TODO: REMOVE THE LAST LABEL

    all_pred_classes = list()
    all_pred_masks = list()
    all_pred_scores = list()
    all_heatmaps = list()
    all_query_pos = list()

    offset_coords_idx = 0
    for bid in range(len(prediction[-1]['pred_masks'])):
        if args.on_segment:
            masks = prediction[-1]['pred_masks'][bid].detach().cpu()[
                target_low_res[bid]['point2segment'].cpu()]
        else:
            masks = prediction[-1]['pred_masks'][bid].detach().cpu()

        if args.use_dbscan:
            new_preds = {
                'pred_masks': list(),
                'pred_logits': list(),
            }

            curr_coords_idx = masks.shape[0]
            curr_coords = raw_coords[offset_coords_idx:curr_coords_idx + offset_coords_idx]
            offset_coords_idx += curr_coords_idx

            for curr_query in range(masks.shape[1]):
                curr_masks = masks[:, curr_query] > 0

                if curr_coords[curr_masks].shape[0] > 0:
                    clusters = DBSCAN(eps=args.dbscan_eps,
                                      min_samples=args.dbscan_min_points,
                                      n_jobs=-1).fit(curr_coords[curr_masks]).labels_

                    new_mask = torch.zeros(curr_masks.shape, dtype=int)
                    new_mask[curr_masks] = torch.from_numpy(clusters) + 1

                    for cluster_id in np.unique(clusters):
                        original_pred_masks = masks[:, curr_query]
                        if cluster_id != -1:
                            new_preds['pred_masks'].append(original_pred_masks * (new_mask == cluster_id + 1))
                            new_preds['pred_logits'].append(
                                prediction[-1]['pred_logits'][bid, curr_query])

            scores, masks, classes, heatmap = get_mask_and_scores(
                args, torch.stack(new_preds['pred_logits']).cpu(),
                torch.stack(new_preds['pred_masks']).T,
                len(new_preds['pred_logits']),
                args.classes - 1)
        else:
            scores, masks, classes, heatmap = get_mask_and_scores(
                args, prediction[-1]['pred_logits'][bid].detach().cpu(),
                masks,
                prediction[-1]['pred_logits'][bid].shape[0],
                args.classes - 1)

        masks = get_full_res_mask(masks, inverse_maps[bid], target_full_res[bid]['point2segment'],
                                  on_segments=args.on_segment)

        heatmap = get_full_res_mask(heatmap, inverse_maps[bid], target_full_res[bid]['point2segment'],
                                    on_segments=args.on_segment, is_heatmap=True)

        if backbone_features is not None:
            backbone_features = get_full_res_mask(torch.from_numpy(backbone_features), inverse_maps[bid],
                                                  target_full_res[bid]['point2segment'], is_heatmap=True)
            backbone_features = backbone_features.numpy()

        masks = masks.numpy()
        heatmap = heatmap.numpy()

        sort_scores = scores.sort(descending=True)
        sort_scores_index = sort_scores.indices.cpu().numpy()
        sort_scores_values = sort_scores.values.cpu().numpy()
        sort_classes = classes[sort_scores_index]

        sorted_masks = masks[:, sort_scores_index]
        sorted_heatmap = heatmap[:, sort_scores_index]

        all_pred_classes.append(sort_classes)
        all_pred_masks.append(sorted_masks)
        all_pred_scores.append(sort_scores_values)
        all_heatmaps.append(sorted_heatmap)

    # if self.validation_dataset.dataset_name == "scannet200":
    #     all_pred_classes[bid][all_pred_classes[bid] == 0] = -1
    #     if self.config.data.test_mode != "test":
    #         target_full_res[bid]['labels'][target_full_res[bid]['labels'] == 0] = -1

    for bid in range(len(prediction[-1]['pred_masks'])):
        all_pred_classes[bid] = remap_model_output(all_pred_classes[bid].cpu() + label_offset)  # map back

        if test_mode != "test" and len(target_full_res) != 0:
            target_full_res[bid]['labels'] = remap_model_output(target_full_res[bid]['labels'].cpu() + label_offset)

        preds[file_names[bid]] = {
            'pred_masks': all_pred_masks[bid],
            'pred_scores': all_pred_scores[bid],
            'pred_classes': all_pred_classes[bid]
        }

        if args.save_visualizations:
            self.save_visualizations(target_full_res[bid],
                                     full_res_coords[bid],
                                     [self.preds[file_names[bid]]['pred_masks']],
                                     [self.preds[file_names[bid]]['pred_classes']],
                                     file_names[bid],
                                     original_colors[bid],
                                     original_normals[bid],
                                     [self.preds[file_names[bid]]['pred_scores']],
                                     sorted_heatmaps=[all_heatmaps[bid]],
                                     query_pos=all_query_pos[bid] if len(all_query_pos) > 0 else None,
                                     backbone_features=backbone_features,
                                     point_size=self.config.general.visualization_point_size)

    return preds


