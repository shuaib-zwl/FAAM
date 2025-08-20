# Copyright (c) OpenMMLab. All rights reserved.
# 小目标数据集会因为标注框过多，时间计算长，导致NCCL报错
# 注意修改MaxIouAssigner的脚本，或者test_cfg关于检测框数量的设置
import torch
import torch.nn as nn
import math
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32
import os
from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from mmcv.cnn import build_norm_layer


def visualize_and_save_feature_maps(feature_map, mask_grid, save_dir):
    """
    展示并保存掩码后的特征图。
    :param feature_map: [1, 16, 1024, 1024] 的 torch.Tensor 或 numpy.ndarray
    :param mask_grid: [1, 1024] 的掩码张量（0/1），每个值代表一个32×32格网是否被保留
    :param save_dir: 保存图片的目标目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu().numpy()
    if isinstance(mask_grid, torch.Tensor):
        mask_grid = mask_grid.detach().cpu().numpy()

    feature_map = feature_map[0]  # 形状: [16, 1024, 1024]
    mask_grid = mask_grid[0]      # 形状: [1024]

    # 将[1024]格网掩码还原到[1024, 1024]空间掩码
    grid_size = 32
    num_grids = 32
    mask_map = np.zeros((1024, 1024), dtype=np.uint8)
    for idx in range(mask_grid.shape[0]):
        row = idx // num_grids
        col = idx % num_grids
        if mask_grid[idx] == 1:
            mask_map[row*grid_size:(row+1)*grid_size, col*grid_size:(col+1)*grid_size] = 1

    # 遍历每个通道进行可视化
    for c in range(feature_map.shape[0]):
        feat = feature_map[c]
        # 掩码为0的位置强制为0，掩码为1的位置保持原始数值
        feat_masked = np.where(mask_map == 1, feat, 0)

        # 构造展示用的图像
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        # 用掩码为1的位置显示viridis色图，掩码为0的位置显示黑色
        img = np.ma.masked_where(mask_map == 0, feat_masked)
        plt.imshow(np.zeros_like(feat), cmap='gray', vmin=0, vmax=feat.max())  # 先显示全黑
        plt.imshow(img, cmap='viridis', vmin=feat_masked.min(), vmax=feat_masked.max())  # 用色图覆盖掩码区域
        plt.title(f'Feature Map Channel {c}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'feature_map_channel_{c}.png'), bbox_inches='tight', pad_inches=0.05, dpi=200)
        plt.close()

    print(f'所有通道特征图已保存至: {save_dir}')


@HEADS.register_module()
class CenterNetAGG(BaseDenseHead, BBoxTestMixin):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_mask=dict(type='BCEWithLogitsLoss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(CenterNetAGG, self).__init__(init_cfg)
        self.num_classes = num_classes
        # 使用 HeadModule 创建各个头
        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_mask = build_loss(loss_mask)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
        # for head in [self.wh_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)  # 获取真实信息
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)  # 计算loss
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, feat):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """

        return multi_apply(self.forward_single, feat)

    def forward_single(self, feat):
        """
        Args:
            feat: 输入特征图，包括静态流和动态流特征图。

        Returns:
            center_heatmap_pred: 优化后的目标中心点预测热图。
            wh_pred: 优化后的宽高预测。
            offset_pred: 优化后的偏移量预测。
        """
        # 调用 MaskProp 进行掩码计算和特征增强
        mask_prop = MaskAttention(grid_size=32, threshold=0.05, embed_dim=16, in_channels=feat.size(1), num_heads=8).to(feat.device)
        agg_feats, mask_pred = mask_prop(feat)
        # save_dir = '/home/ubuntu/data/ZWL/ZWL/data/pic/feature_maps_mask/0.png'

        # visualize_and_save_feature_maps(agg_feats, mask_pred, save_dir)


        # visualize_mask(mask_pred)
        center_heatmap_pred = self.heatmap_head(agg_feats).sigmoid()
        wh_pred = self.wh_head(agg_feats)
        offset_pred = self.offset_head(agg_feats)

        return center_heatmap_pred, wh_pred, offset_pred, mask_pred
        # return center_heatmap_pred, wh_pred, mask_pred

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             mask_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        # assert len(center_heatmap_preds) == len(wh_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]

        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                     center_heatmap_pred.shape,
                                                     img_metas[0]['pad_shape'], wh_pred)

        # get_gt_mask
        bs, _, W, H = wh_pred.shape
        gt_mask = self.get_gt_mask(gt_bboxes, grid_size=32, feature_map_size=(bs, H, W)).view(bs, -1)
        mask_preds = mask_preds[0]

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        offset_target_weight = target_result['offset_target_weight']  # 说明哪些区域目标值用于计算
        wh_target_weight = target_result['wh_target_weight']  # 说明哪些区域目标值用于计算
        # wh_target_weight = wh_target_weight[:, 0, :, :]
        # a = (wh_target_weight > 0).sum()

        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        loss_wh = self.loss_wh(
            wh_pred,
            wh_target,
            wh_target_weight,
            avg_factor=avg_factor * 8)
        loss_offset = self.loss_offset(offset_pred, offset_target, offset_target_weight, avg_factor=avg_factor * 8)
        loss_mask = self.loss_mask(mask_preds, gt_mask)
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_mask=loss_mask)

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape, wh_pred):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.
            wh_pred (list[int]): pred_list.
        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target_weight = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_target_weight = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w])

        device = gt_bboxes[0].device
        # 编辑的预设信息
        topk = 25
        rf_radius = 3  # 感受野视野
        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            # 缩放到特征图尺度
            gt_scale_bbox = gt_bbox * torch.tensor([width_ratio, height_ratio, width_ratio, height_ratio],
                                                   device=device)

            for gt_idx, (bbox, label) in enumerate(zip(gt_scale_bbox, gt_label)):
                gt_xmin, gt_ymin, gt_xmax, gt_ymax = bbox.int()

                center_x = ((gt_xmin + gt_xmax) / 2).long()
                center_y = ((gt_ymin + gt_ymax) / 2).long()
                ori_center_x = ((gt_xmin + gt_xmax) / 2)
                ori_center_y = ((gt_ymin + gt_ymax) / 2)

                gt_xmin = torch.clamp(gt_xmin, 0, feat_w - 1)
                gt_xmax = torch.clamp(gt_xmax, 0, feat_w - 1)
                gt_ymin = torch.clamp(gt_ymin, 0, feat_h - 1)
                gt_ymax = torch.clamp(gt_ymax, 0, feat_h - 1)

                x_range = torch.arange(gt_xmin, gt_xmax + 1, device=device)
                y_range = torch.arange(gt_ymin, gt_ymax + 1, device=device)

                yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
                points_coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

                # 每个像素感受野bbox
                points_bbox = torch.cat([points_coords - rf_radius, points_coords + rf_radius], dim=-1)

                # 当前gt的高斯bbox表示
                gt_bbox_gaussian = bbox.unsqueeze(0)

                # 计算每个gt与gt内坐标点的KL散度
                kl_scores = bbox_overlaps(
                    gt_bbox_gaussian.unsqueeze(0),  # [1,1,4]
                    points_bbox.unsqueeze(0),  # [1,num_points,4]
                    mode='kl').view(-1)  # [num_points]

                # 获取KL散度前topk的点
                if kl_scores.numel() >= topk:
                    topk_scores, topk_indices = torch.topk(kl_scores, topk)
                else:
                    topk_scores, topk_indices = kl_scores, torch.arange(kl_scores.size(0), device=device)

                # 初次筛选正样本坐标
                positive_coords = points_coords[topk_indices]

                # 获取wh_pred对应的预测坐标及宽高

                # wh_pred_x = wh_pred[batch_id, 0, positive_coords[:, 1], positive_coords[:, 0]]
                # wh_pred_y = wh_pred[batch_id, 1, positive_coords[:, 1], positive_coords[:, 0]]
                wh_pred_w = wh_pred[batch_id, 0, positive_coords[:, 1], positive_coords[:, 0]]
                wh_pred_h = wh_pred[batch_id, 1, positive_coords[:, 1], positive_coords[:, 0]]

                # 更新wh目标
                gt_w = (bbox[2] - bbox[0]).item()
                gt_h = (bbox[3] - bbox[1]).item()

                # 如果IOU大于阈值，则认为是精细筛选正样本
                iou_threshold = 0.7
                iou = compute_iou(wh_pred_w, wh_pred_h, gt_w, gt_h)
                valid_coords = positive_coords[iou > iou_threshold]

                # 如果有有效的正样本坐标，更新wh_target,heatmap_target
                if valid_coords.numel() > 0:
                    # 2.1 取出 valid_coords 的 x、y，并转成浮点
                    vx = valid_coords[:, 0].float()  # 列坐标
                    vy = valid_coords[:, 1].float()  # 行坐标

                    # 2.2 计算 l^*, r^*, t^*, b^*
                    l_star = vx - gt_xmin
                    r_star = gt_xmax - vx
                    t_star = vy - gt_ymin
                    b_star = gt_ymax - vy

                    # 2.4 使用 FCOS 原论文中的 centerness 公式
                    lr_min = torch.min(l_star, r_star)
                    lr_max = torch.max(l_star, r_star)
                    tb_min = torch.min(t_star, b_star)
                    tb_max = torch.max(t_star, b_star)
                    centerness_values = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max))

                    wh_target[batch_id, 0, valid_coords[:, 1], valid_coords[:, 0]] = gt_w
                    wh_target[batch_id, 1, valid_coords[:, 1], valid_coords[:, 0]] = gt_h
                    offset_target[batch_id, 0, valid_coords[:, 1], valid_coords[:, 0]] = ori_center_x - valid_coords[:, 0]
                    offset_target[batch_id, 1, valid_coords[:, 1], valid_coords[:, 0]] = ori_center_y - valid_coords[:, 1]
                    # print((offset_target[:,0,:,:]>1).sum())
                    wh_target_weight[batch_id, :, valid_coords[:, 1], valid_coords[:, 0]] = centerness_values   # yx的格式
                    offset_target_weight[batch_id, :, valid_coords[:, 1], valid_coords[:, 0]] = centerness_values   # yx的格式
                    offset_target_weight[batch_id, :, valid_coords[:, 1], valid_coords[:, 0]] = 1   # yx的格式
                    center_heatmap_target[batch_id, label, valid_coords[:, 1], valid_coords[:, 0]] = centerness_values
                    # radius = gaussian_radius([gt_h, gt_w], min_overlap=0.3)
                    # radius = max(0, int(radius))
                    # ind = gt_label[gt_idx]
                    # gen_gaussian_target(center_heatmap_target[batch_id, ind], [center_x, center_y], radius)
                else:
                    # 没有有效正样本，使用gaussian radius计算
                    radius = gaussian_radius([gt_h, gt_w], min_overlap=0.3)
                    radius = max(0, int(radius))
                    ind = gt_label[gt_idx]
                    gen_gaussian_target(center_heatmap_target[batch_id, ind], [center_x, center_y], radius)
                    wh_target[batch_id, 0, center_y, center_x] = gt_w
                    wh_target[batch_id, 1, center_y, center_x] = gt_h
                    offset_target[batch_id, 0, center_y, center_x] = ori_center_x - center_x
                    offset_target[batch_id, 1, center_y, center_x] = ori_center_y - center_y
                    wh_target_weight[batch_id, :, center_y, center_x] = 1
                    offset_target_weight[batch_id, :, center_y, center_x] = 1
                # 如果wh位置不准确，提高中心点的权重，使用下面代替上面
                # wh_target_weight[batch_id, :, center_y, center_x] = 1
                center_heatmap_target[batch_id, label, center_y, center_x] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            wh_target_weight=wh_target_weight,
            offset_target=offset_target,
            offset_target_weight=offset_target_weight)
        return target_result, avg_factor

    def get_gt_mask(self, gt_bboxes, grid_size, feature_map_size):
        B, H, W = feature_map_size  # B = batch size, H = height, W = width

        # 计算格网的行数和列数
        num_grids_h = H // grid_size
        num_grids_w = W // grid_size

        # 初始化 GT Mask，初始全为 0
        gt_mask = torch.zeros((B, num_grids_h, num_grids_w), device=gt_bboxes[0].device)

        for i in range(B):
            # 获取每个 batch 对应的 gt_bboxes
            bbox_tensor = gt_bboxes[i]  # 这是一个 [N, 4] 的 tensor，N 是目标数量
            for bbox in bbox_tensor:  # 每个目标的边界框
                xmin, ymin, xmax, ymax = bbox
                # 计算该目标所在的格网的坐标
                xmin_grid = int(xmin // grid_size)
                ymin_grid = int(ymin // grid_size)
                xmax_grid = int(xmax // grid_size)
                ymax_grid = int(ymax // grid_size)

                # 将这些格网区域标记为有目标（1）
                gt_mask[i, ymin_grid:ymax_grid + 1, xmin_grid:xmax_grid + 1] = 1

        return gt_mask

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   mask_preds,
                   img_metas,
                   rescale=True,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): WH predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        # assert len(center_heatmap_preds) == len(wh_preds) == 1
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    center_heatmap_preds[0][img_id:img_id + 1, ...],
                    wh_preds[0][img_id:img_id + 1, ...],
                    offset_preds[0][img_id:img_id + 1, ...],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))
        return result_list

    def _get_bboxes_single(self,
                           center_heatmap_pred,
                           wh_pred,
                           offset_pred,
                           img_meta,
                           rescale=False,
                           with_nms=True):
        """Transform outputs of a single image into bbox results.

        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            wh_pred (Tensor): WH heatmap for current level with shape
                (1, num_classes, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: The first item is an (n, 5) tensor, where
                5 represent (tl_x, tl_y, br_x, br_y, score) and the score
                between 0 and 1. The shape of the second tensor in the tuple
                is (n,), and each element represents the class label of the
                corresponding box.
        """
        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            img_meta['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)

        # batch_border = det_bboxes.new_tensor(img_meta['border'])[...,
        #                                                         [2, 0, 2, 0]]
        # det_bboxes[..., :4] -= batch_border

        if rescale:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(
                img_meta['scale_factor'])

        if with_nms:
            det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels,
                                                      self.test_cfg)
        return det_bboxes, det_labels

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        # center_heatmap_pred = get_local_maximum(center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets
        # visualize_topk(topk_xs,topk_ys)
        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        # visualize_wh_and_offset(wh, offset)
        # print((offset > 1).sum())
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        # dup_cells, dup_indices = find_duplicate_centers_by_intcell(topk_xs, topk_ys)
        '''        
        if dup_indices:
            print("发现以下整数格子内有多个点：")
            for cell, inds in zip(dup_cells.tolist(), dup_indices):
                print(f"  格子 {cell} 包含索引 {inds}（共 {len(inds)} 个点）")
        else:
            print("所有点的整数部分都各不相同。")'''

        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1].contiguous(), labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]

        return bboxes, labels

def visualize_wh_and_offset(wh: torch.Tensor,
                            offset: torch.Tensor,
                            point_size=20,
                            alpha=0.6):
    """
    输入：
      wh:     Tensor, shape (1, N, 2)，表示 N 个框的 (w, h)
      offset: Tensor, shape (1, N, 2)，表示 N 个框的 (dx, dy)
    会绘制两张图：
      1) wh：宽-高 散点图
      2) offset：dx-dy 散点图
    """
    # 展平并转 numpy
    wh_np = wh[0].cpu().numpy()           # (N,2)
    w = wh_np[:, 0]
    h = wh_np[:, 1]

    off_np = offset[0].cpu().numpy()      # (N,2)
    dx = off_np[:, 0]
    dy = off_np[:, 1]

    # 1) WH 分布
    plt.figure(figsize=(5,5))
    plt.scatter(w, h, s=point_size, alpha=alpha)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Distribution of Width vs Height")
    plt.grid(False)
    plt.show()

    # 2) Offset 分布
    plt.figure(figsize=(5,5))
    plt.scatter(dx, dy, s=point_size, alpha=alpha)
    plt.xlabel("Offset X")
    plt.ylabel("Offset Y")
    plt.title("Distribution of Offset X vs Offset Y")
    plt.axhline(0, linestyle='--', linewidth=0.5)
    plt.axvline(0, linestyle='--', linewidth=0.5)
    plt.grid(False)
    plt.show()


def visualize_topk(topk_xs: torch.Tensor,
                   topk_ys: torch.Tensor,
                   map_size=(1024, 1024),
                   point_size=20,
                   alpha=0.6):
    """
    将 topk 坐标画到一张 map_size 大小的图上

    Args:
        topk_xs (Tensor): (1, N) tensor，表示所有点的 x 坐标
        topk_ys (Tensor): (1, N) tensor，表示所有点的 y 坐标
        map_size (tuple): (宽, 高)，这里设为 (1024,1024)
        point_size (int): 散点大小
        alpha (float): 散点透明度
    """
    # 展平并转为 numpy
    xs = topk_xs.view(-1).cpu().numpy()
    ys = topk_ys.view(-1).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=point_size, alpha=alpha)
    plt.xlim(0, map_size[0])
    plt.ylim(0, map_size[1])
    # 图像坐标系常以左上为原点，Y 轴向下增长，用 invert_yaxis 翻转
    plt.gca().invert_yaxis()
    plt.title("Top-k Points Distribution on 1024×1024 Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)
    plt.show()


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (B, num_windows, C, window_size, window_size)
    """
    B, C, H, W = x.shape  # 获取批次大小、通道数、特征图的高宽
    assert H % window_size == 0 and W % window_size == 0, "Feature map size must be divisible by window size"

    # 将特征图按照窗口大小切分，分割为多个窗口
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)

    # 重新排列维度，并展平，使得每个窗口成为一个独立的样本
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, -1, C, window_size, window_size)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (B, num_windows, C, window_size, window_size)
        window_size (int): window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B, num_windows, C, window_size, window_size = windows.shape
    assert H % window_size == 0 and W % window_size == 0, "Feature map size must be divisible by window size"

    # 计算出每个维度上可以容纳多少个窗口
    h_strides = H // window_size
    w_strides = W // window_size

    # 恢复成原来的特征图
    x = windows.view(B, h_strides, w_strides, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)  # 拼接并恢复到 [B, C, H, W]

    return x



def visualize_mask(mask, grid_size=32, save_path=None):
    """
    Visualize mask values as a grid of 32x32 with white for 0 and black for 1,
    and optionally save the figure to a specified directory.

    Args:
        mask (Tensor): Mask tensor of shape (B, number_grid, 1, 1, 1)
        grid_size (int): Size of the grid (default is 32)
        save_path (str or None): 如果提供，将图像保存到该路径；否则直接显示
    """
    # Assume mask shape is (1, 1024, 1, 1, 1) with batch size B=1
    B, number_grid, _, _, _ = mask.shape

    # Convert mask from (B, number_grid, 1, 1, 1) to (B, number_grid)
    mask = mask.view(B, number_grid).squeeze()

    # Reshape the mask to a 32x32 grid
    mask_grid = mask.view(grid_size, grid_size).cpu().numpy()  # Convert to numpy for visualization

    # Plotting the mask grid
    plt.figure(figsize=(6, 6))
    plt.imshow(mask_grid, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title("Mask Visualization")

    if save_path:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved mask visualization to: {save_path}")
    else:
        plt.show()


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    assert mode in ['iou', 'iof', 'giou', 'wd', 'kl', 'center_distance2', 'exp_kl', 'kl_10'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)

    if rows * cols == 0:
        return bboxes1.new(batch_shape + (rows, cols))

    if mode == 'kl':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps
        # (num_gt, num_feat_point)
        kl = (w2 ** 2 / w1 ** 2 + h2 ** 2 / h1 ** 2 + 4 * whs[..., 0] ** 2 / w1 ** 2 + 4 * whs[
            ..., 1] ** 2 / h1 ** 2 + torch.log(w1 ** 2 / w2 ** 2) + torch.log(h1 ** 2 / h2 ** 2) - 2) / 2

        kld = 1 / (1 + kl)

        return kld



def compute_iou(pred_w, pred_h, gt_w, gt_h):
    """计算预测宽高与真实宽高之间的IoU"""
    # 将gt_w和gt_h转换为与pred_w和pred_h相同类型的张量
    gt_w = torch.tensor(gt_w, dtype=pred_w.dtype, device=pred_w.device)
    gt_h = torch.tensor(gt_h, dtype=pred_h.dtype, device=pred_h.device)

    inter_area = torch.min(pred_w, gt_w) * torch.min(pred_h, gt_h)
    pred_area = pred_w * pred_h
    gt_area = gt_w * gt_h
    union_area = pred_area + gt_area - inter_area
    iou = inter_area / union_area
    return iou



class ObjSeekerWithDilatedConv(nn.Module):
    def __init__(self, in_channels=16, dilation_rate=2, bias_init=-4.6):
        """
        Args:
            in_channels (int): 输入特征图的通道数 (默认: 16)。
            dilation_rate (int): 空洞卷积的膨胀率 (默认: 2)。
            bias_init (float): 1×1 卷积偏置的初始值 (默认: -4.6)。
        """
        super(ObjSeekerWithDilatedConv, self).__init__()

        # 空洞卷积 (感受野通过膨胀扩展为 5×5 或更大)
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1,
            padding=dilation_rate, dilation=dilation_rate, groups=in_channels, bias=False)

        # 批量归一化和非线性激活
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # 最终通过 1×1 卷积将特征图压缩为单通道掩码
        self.conv = nn.Conv2d(
            in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

        # 初始化偏置，初始掩码接近 0
        self.conv.bias.data.fill_(bias_init)

    def forward(self, feat):
        """
        Args:
            feat (Tensor): 输入特征图 (B, C, H, W)

        Returns:
            mask (Tensor): 生成的掩码 (B, 1, H, W)，范围 [0, 1]
        """
        # 深度空洞卷积，感受野扩大
        x = self.dwconv(feat)

        # 批量归一化和激活
        x = self.bn(x)
        x = self.relu(x)

        # 最终生成掩码
        mask = torch.sigmoid(self.conv(x))
        return mask


class MaskAttention(nn.Module):
    def __init__(self, grid_size, threshold, in_channels, embed_dim, num_heads, dropout: float = 0.0):
        super(MaskAttention, self).__init__()
        self.grid_size = grid_size  # 窗口大小（Window Size）
        self.threshold = threshold  # 掩码筛选的阈值
        self.obj_seeker = ObjSeekerWithDilatedConv(in_channels=in_channels)

        self.proj = nn.Linear(in_channels, embed_dim)
        #   使用 nn.Parameter 将位置编码设为可学习参数（也可以使用固定的正弦位置编码）
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, grid_size, grid_size))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.scale = embed_dim ** -0.5  # 归一化因子

    def forward(self, feat):
        dy_feat = feat
        B, C, H, W = dy_feat.shape
        # 掩码生成
        device = feat.device
        initial_mask = self.obj_seeker(dy_feat)  # B, 1, H, W

        # ----------- 划分特征图为格网（窗口） -----------
        grid_size = self.grid_size
        dy_feat_patch = window_partition(dy_feat, grid_size)  # Wn, C, grid_size, grid_size
        mask_patch = window_partition(initial_mask, grid_size)
        _, Wn, _, _, _ = dy_feat_patch.shape  # Wn: 总窗口数

        # ----------- 计算掩码（基于格网最大值） -----------
        mask_info = mask_patch.view(B, Wn, -1).max(dim=-1)  # Wn, 1
        mask_preds = mask_info[0]
        binary_mask = (mask_preds > self.threshold).float().view(B, Wn, 1, 1, 1)  # B, Wn, 1, 1, 1
        # visualize_mask(binary_mask,save_path="/data/ZWL/ZWL/data/pic/mask.png")
        grid_indices = (binary_mask.view(B, -1) == 1).nonzero()

        masked_feat_patch = dy_feat_patch * binary_mask
        enhanced_feat = window_reverse(masked_feat_patch, grid_size, H, W)  # B, C, H, W

        # visualize_feature_maps(enhanced_feat.unsqueeze(1))
        return enhanced_feat, mask_preds

