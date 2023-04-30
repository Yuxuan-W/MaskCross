import torch
import numpy as np


def save_visualizations(target_full, full_res_coords,
                        sorted_masks, sort_classes, file_name, original_colors, original_normals,
                        sort_scores_values, point_size=20, sorted_heatmaps=None,
                        query_pos=None, backbone_features=None):

    full_res_coords -= full_res_coords.mean(axis=0)

    gt_pcd_pos = []
    gt_pcd_normals = []
    gt_pcd_color = []
    gt_inst_pcd_color = []
    gt_boxes = []

    if 'labels' in target_full:
        instances_colors = torch.from_numpy(
            np.vstack(get_evenly_distributed_colors(target_full['labels'].shape[0])))
        for instance_counter, (label, mask) in enumerate(zip(target_full['labels'], target_full['masks'])):
            if label == 255:
                continue

            mask_tmp = mask.detach().cpu().numpy()
            mask_coords = full_res_coords[mask_tmp.astype(bool), :]

            if len(mask_coords) == 0:
                continue

            gt_pcd_pos.append(mask_coords)
            mask_coords_min = full_res_coords[mask_tmp.astype(bool), :].min(axis=0)
            mask_coords_max = full_res_coords[mask_tmp.astype(bool), :].max(axis=0)
            size = mask_coords_max - mask_coords_min
            mask_coords_middle = mask_coords_min + size / 2

            gt_boxes.append({"position": mask_coords_middle, "size": size,
                             "color": self.validation_dataset.map2color([label])[0]})

            gt_pcd_color.append(
                self.validation_dataset.map2color([label]).repeat(gt_pcd_pos[-1].shape[0], 1)
            )
            gt_inst_pcd_color.append \
                (instances_colors[instance_counter % len(instances_colors)].unsqueeze(0).repeat(gt_pcd_pos[-1].shape[0], 1))

            gt_pcd_normals.append(original_normals[mask_tmp.astype(bool), :])

        gt_pcd_pos = np.concatenate(gt_pcd_pos)
        gt_pcd_normals = np.concatenate(gt_pcd_normals)
        gt_pcd_color = np.concatenate(gt_pcd_color)
        gt_inst_pcd_color = np.concatenate(gt_inst_pcd_color)

    v = vis.Visualizer()

    v.add_points("RGB Input", full_res_coords,
                 colors=original_colors,
                 normals=original_normals,
                 visible=True,
                 point_size=point_size)

    if backbone_features is not None:
        v.add_points("PCA", full_res_coords,
                     colors=backbone_features,
                     normals=original_normals,
                     visible=False,
                     point_size=point_size)

    if 'labels' in target_full:
        v.add_points("Semantics (GT)", gt_pcd_pos,
                     colors=gt_pcd_color,
                     normals=gt_pcd_normals,
                     alpha=0.8,
                     visible=False,
                     point_size=point_size)
        v.add_points("Instances (GT)", gt_pcd_pos,
                     colors=gt_inst_pcd_color,
                     normals=gt_pcd_normals,
                     alpha=0.8,
                     visible=False,
                     point_size=point_size)

    pred_coords = []
    pred_normals = []
    pred_sem_color = []
    pred_inst_color = []

    for did in range(len(sorted_masks)):
        instances_colors = torch.from_numpy(
            np.vstack(get_evenly_distributed_colors(max(1, sorted_masks[did].shape[1]))))

        for i in reversed(range(sorted_masks[did].shape[1])):
            coords = full_res_coords[sorted_masks[did][:, i].astype(bool), :]

            mask_coords = full_res_coords[sorted_masks[did][: ,i].astype(bool), :]
            mask_normals = original_normals[sorted_masks[did][: ,i].astype(bool), :]

            label = sort_classes[did][i]

            if len(mask_coords) == 0:
                continue

            pred_coords.append(mask_coords)
            pred_normals.append(mask_normals)

            pred_sem_color.append(
                self.validation_dataset.map2color([label]).repeat(
                    mask_coords.shape[0], 1)
            )

            pred_inst_color.append(instances_colors[i % len(instances_colors)].unsqueeze(0).repeat(
                mask_coords.shape[0], 1)
            )

        if len(pred_coords) > 0:
            pred_coords = np.concatenate(pred_coords)
            pred_normals = np.concatenate(pred_normals)
            pred_sem_color = np.concatenate(pred_sem_color)
            pred_inst_color = np.concatenate(pred_inst_color)

            v.add_points("Semantics (Mask3D)", pred_coords,
                         colors=pred_sem_color,
                         normals=pred_normals,
                         visible=False,
                         alpha=0.8,
                         point_size=point_size)
            v.add_points("Instances (Mask3D)", pred_coords,
                         colors=pred_inst_color,
                         normals=pred_normals,
                         visible=False,
                         alpha=0.8,
                         point_size=point_size)

    v.save(f"{self.config['general']['save_dir']}/visualizations/{file_name}")