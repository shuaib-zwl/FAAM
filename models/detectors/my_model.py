# Copyright (c) OpenMMLab. All rights reserved.
import torch
import matplotlib.pyplot as plt
from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from mmdet.core.utils import flip_tensor
from mmdet.models.detectors.single_stage import SingleStageDetector
import cv2
import numpy as np
from mmdet.core import multiclass_nms
import torch.nn.functional as F


# 假设你有 img_clu，这是原始图像和掩码相乘得到的结果
# img_clu 是 (Batch, Channel, Height, Width) 的张量
def adjust_scale(cropped_width, cropped_height):
    # 计算放大后尺寸满足1024倍数的最小scale_factor
    target_width = (cropped_width +1023) // 1024 * 1024
    target_height = (cropped_height + 1023) // 1024 * 1024

    # 计算放大因子
    scale_factor_width = target_width / cropped_width
    scale_factor_height = target_height / cropped_height

    # 选择较大的放大因子以确保两边都满足要求
    scale_factor = max(scale_factor_width, scale_factor_height)

    return scale_factor


# 实现nms

def is_inside_cluster(box, cluster_bboxes):
    """
    判断一个检测框是否完全位于任意簇区域内
    :param box: 单个检测框 [x_min, y_min, x_max, y_max]
    :param cluster_bboxes: 簇区域列表 [[x_min, y_min, x_max, y_max], ...]
    :return: True (在任意簇区域内), False (不在任何簇区域内)
    """
    for cluster in cluster_bboxes:
        # 检查检测框是否完全位于簇区域内
        if (box[0] >= cluster[0] and box[1] >= cluster[1] and
                box[2] <= cluster[2] and box[3] <= cluster[3]):
            return True  # 检测框在簇区域内
    return False

def filter_bboxes_by_clusters(original_bboxes, cluster_bboxes):
    """
    剔除原始检测框中位于簇区域内的框
    :param original_bboxes: 原始检测框列表 [[x_min, y_min, x_max, y_max, score], ...]
    :param cluster_bboxes: 簇区域边界框列表 [[x_min, y_min, x_max, y_max], ...]
    :return: 过滤后的检测框列表
    """
    filtered_bboxes = []
    for box in original_bboxes:
        if not is_inside_cluster(box[:4], cluster_bboxes):  # 检查检测框是否在簇内
            filtered_bboxes.append(box)  # 保留不在簇内的框
    return np.array(filtered_bboxes)

def generate_img_metas(original_img, img_crop, img_crop_resized, scale_factor=[1.0, 1.0, 1.0, 1.0], flip=False, flip_direction=None):
    # original_img: 原始图像
    # img_crop: 裁剪后的图像
    ori_shape = original_img.permute(0,2,3,1).shape[1:4]  # 原始图像的大小
    img_shape = img_crop.permute(0,2,3,1).shape[1:4]  # 裁剪后图像的大小
    pad_shape = img_crop_resized.permute(0,2,3,1).shape[1:4]   # 假设裁剪后没有额外的填充

    img_metas = [{
        'filename': 'cropped_image.jpg',  # 可以设置为裁剪后的文件名或路径
        'ori_shape': tuple(ori_shape),  # 原始图像大小
        'img_shape': tuple(img_shape),  # 裁剪后图像大小
        'pad_shape': tuple(pad_shape),  # 如果有填充，记录填充后的大小
        'scale_factor': np.array([scale_factor, scale_factor, scale_factor, scale_factor], dtype=np.float32),  # 缩放因子
        'flip': flip,  # 是否翻转
        'flip_direction': flip_direction,  # 翻转方向
        'img_norm_cfg': {  # 图像归一化参数
            'mean': np.array([105.59, 103.17, 99.6], dtype=np.float32),  # 归一化的均值
            'std': np.array([39.88, 43.1, 46.53], dtype=np.float32),  # 归一化的标准差
            'to_rgb': True
        },
        'border': np.array([0, 0, 0, 0], dtype=np.float32),
        'batch_input_shape': pad_shape[:2]  # 输入到模型的图像大小（通常与pad_shape相同）
    }]

    return img_metas

# 粘贴的centernet
@DETECTORS.register_module()
class MYmodel(SingleStageDetector):
    """Implementation of YOLC

    <https://arxiv.org/abs/2404.06180>.
    """
    #
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
         super(MYmodel, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)

    # 执行测试，提取特征图然后调用bbox—head.simple test获取预测结果，进行裁剪/直接返回检测框的结果
    def simple_test(self, img, img_metas, rescale=True, crop=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).（批次，通道，宽高）
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # 输入特征图维度[batchsize,384,264,264]
        feat = self.extract_feat(img)
        if crop:
            # return bbox_results
            maxcontours, results_list = self.bbox_head.simple_test(
                feat, img_metas, rescale=rescale, crop=crop)

            # 源图像区域裁剪
            _, _, height, width = img.shape
            # 设置使用的簇的个数
            K = 1
            # K = maxcontours.shape[0]
            areas = [contour[2] * contour[3] for contour in maxcontours]
            sorted_indices = np.argsort(areas)[::-1]  # 从大到小排序
            top_k_indices = sorted_indices[:K]  # 取前 K 大的索引
            # 初始化用于存储所有检测框和标签的列表
            all_det_bboxes_list = []
            all_det_labels_list = []
            cluster_bboxes = []
            # 对每个簇进行处理
            for idx in top_k_indices:
                contour = maxcontours[idx]
                x, y, w, h = contour.astype(int)  # maxcontours 是 xywh 格式

                # 计算右下角的坐标 (x2, y2)
                x1, y1 = x, y
                x2 = x1 + w
                y2 = y1 + h
                cluster_bboxes.append([x1, y1, x2, y2])
                # 使用 torch.clamp 确保坐标值在合法范围内
                x1 = torch.clamp(torch.tensor(x1), 0, width - 1).item()
                y1 = torch.clamp(torch.tensor(y1), 0, height - 1).item()
                x2 = torch.clamp(torch.tensor(x2), 0, width - 1).item()
                y2 = torch.clamp(torch.tensor(y2), 0, height - 1).item()
                # visualize_tensor_image(img, title="Random Image")
                # 从原始图像中裁剪簇对应的区域
                img_crop = img[:, :, y1:y2, x1:x2]

                # visualize_tensor_image(img_crop, title="Random Image")
                # 获取裁剪区域的宽度和高度
                crop_height, crop_width = img_crop.shape[2], img_crop.shape[3]

                # 填充
                # 初始化填充变量为 0
                pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
                # 如果宽度和高度不相等，对短边进行对称填充
                if crop_height != crop_width:
                    if crop_height > crop_width:
                        # 计算需要填充的像素数，使得宽度等于高度
                        pad_left = int((crop_height - crop_width) / 2)
                        pad_right = crop_height - crop_width - pad_left

                        # 对宽度进行填充
                        img_crop_padded = F.pad(img_crop, (pad_left, pad_right, 0, 0), mode='constant', value=0)
                    else:
                        # 计算需要填充的像素数，使得高度等于宽度
                        pad_top = int((crop_width - crop_height) / 2)
                        pad_bottom = crop_width - crop_height - pad_top
                        # 对高度进行填充
                        img_crop_padded = F.pad(img_crop, (0, 0, pad_top, pad_bottom), mode='constant', value=0)
                else:
                    # 如果宽高已经相等，不需要填充
                    img_crop_padded = img_crop

                # 获取调整后的scale_factor
                scale_factor = adjust_scale(img_crop_padded.shape[2], img_crop_padded.shape[3])
                new_height = int(img_crop_padded.shape[2] * scale_factor)
                new_width = int(img_crop_padded.shape[3] * scale_factor)
                # visualize_tensor_image(img_crop, title="Random Image")
                # 使用插值调整放大后的图像大小
                img_crop_resized = F.interpolate(
                    img_crop_padded, size=(new_height, new_width), mode='bilinear', align_corners=True
                )

                # 将裁剪后的图像簇送入特征提取器
                feat_crop = self.extract_feat(img_crop_resized)
                # 使用bbox_head进行检测，得到当前簇的检测结果
                original_img = img
                pad_shape = img_crop_resized
                # 生成 img_metas
                new_img_metas = generate_img_metas(original_img, img_crop, pad_shape, scale_factor=scale_factor)

                results_list_crop = self.bbox_head.simple_test(
                    feat_crop, new_img_metas, rescale=False)

                # 将簇中的检测框坐标还原为原始图像中的坐标
                det_bboxes = results_list_crop[0][0]  # 取出检测框 (tensor)
                det_labels = results_list_crop[0][1]  # 取出检测标签 (tensor)
                # 观察边界框调整的辅助函数
                # visualize_image_with_boxes2(img, det_bboxes.cpu().numpy(), title="ori_img with det box")
                # 缩小检测框坐标，将放大的坐标还原到原来的比例
                det_bboxes[:, :4] /= scale_factor  # 对前四个值 (x_min, y_min, x_max, y_max) 进行缩小
                # 去除填充
                det_bboxes[:, 0] -= pad_left
                det_bboxes[:, 1] -= pad_top
                det_bboxes[:, 2] -= pad_left
                det_bboxes[:, 3] -= pad_top

                # 这里如果检测结果不正确，需要观察设置偏移,
                # 平移得到原始坐标，将检测框还原到原始图像的坐标系中
                det_bboxes[:, 0] += x1
                det_bboxes[:, 1] += y1
                det_bboxes[:, 2] += x1
                det_bboxes[:, 3] += y1
                # 观察边界框调整的辅助函数
                # visualize_image_with_boxes2(img, det_bboxes.cpu().numpy(), title="ori_img with adjusted box")
                # 将当前簇的检测框和标签添加到列表中
                all_det_bboxes_list.append(det_bboxes)
                all_det_labels_list.append(det_labels)

            # 在处理完所有簇后，拼接所有的检测框和标签
            if all_det_bboxes_list:
                all_det_bboxes = torch.cat(all_det_bboxes_list, dim=0)
                all_det_labels = torch.cat(all_det_labels_list, dim=0)
            else:
                all_det_bboxes = torch.empty((0, 5))  # 假设检测框有5个元素（包括得分）
                all_det_labels = torch.empty((0,), dtype=torch.long)
            final_detections = [(all_det_bboxes, all_det_labels)]
            # visualize_image_with_boxes2(img, all_det_bboxes.cpu().numpy(), title="ori_img with adjusted box")

            # 用来将边界框信息按照类别标签进行分类，如果是单类基本没什么变化
            bbox_results_clu = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in final_detections]
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in results_list]

            det_bboxes_ori = bbox_results[0][0]  # 从嵌套列表中提取边界框
            det_bboxes_clu = bbox_results_clu[0][0]
            filtered_bboxes = filter_bboxes_by_clusters(det_bboxes_ori, cluster_bboxes)

            # 确保非簇区域检测框和簇区域检测框是二维数组
            if det_bboxes_clu.size == 0:
                det_bboxes_clu = np.empty((0, 5))  # 如果为空，初始化为空二维数组
            if filtered_bboxes.size == 0:
                filtered_bboxes = np.empty((0, 5))  # 如果为空，初始化为空二维数组

            # 计算非簇区域坐标格式以及簇区域需要保留的坐标个数
            num_non_cluster_bboxes = len(filtered_bboxes)
            num_to_keep_from_clusters = max(0, det_bboxes_clu.shape[0] - num_non_cluster_bboxes)
            # 按得分对簇区域检测框排序并剔除一些检测框
            sorted_bboxes = det_bboxes_clu[det_bboxes_clu[:, 4].argsort()[::-1]]
            top_k_bboxes = sorted_bboxes[:num_to_keep_from_clusters]

            # 合并两个分支的边界框
            det_bboxes = np.concatenate((np.array(filtered_bboxes), np.array(det_bboxes_clu)), axis=0)
            # 将边界框从 numpy 转换为 tensor
            det_bboxes_tensor = torch.from_numpy(det_bboxes).float().to(img.device)

            # 获取边界框的坐标 (x1, y1, x2, y2) 和得分 (score)
            bboxes = det_bboxes_tensor[:, :4]
            scores = det_bboxes_tensor[:, 4]
            # 将 bboxes 和 scores 结合为一个 tensor, 确保 bboxes 包含置信度得分
            bboxes_with_scores = torch.cat([bboxes, scores[:, None]], dim=1)

            # 将最终结果转换回 numpy 格式，并保持 bbox_results 的嵌套结构一致
            bbox_results_last = [[bboxes_with_scores.cpu().numpy()]]  # 确保是两层嵌套
            return bbox_results_last
            # 返回最终的合并、NMS、筛选后的检测结果

        else:  # 输入特征图，获取检测结果，通过bbox_head.simple_test，过完了整个yolc_head.py文档
            results_list = self.bbox_head.simple_test(
                feat, img_metas, rescale=rescale, crop=crop)
            # 用来将边界框信息按照类别标签进行分类，如果是单类基本没什么变化
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in results_list]
            return bbox_results


    # 粘贴的centernet
    # 将数据增强后的多个检测结果进行合并，使用nms去除冗余框
    def merge_aug_results(self, aug_results, with_nms):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            with_nms (bool): If True, do nms before return boxes.

        Returns:
            tuple: (out_bboxes, out_labels)
        """
        recovered_bboxes, aug_labels = [], []
        for single_result in aug_results:
            recovered_bboxes.append(single_result[0][0])
            aug_labels.append(single_result[0][1])

        bboxes = torch.cat(recovered_bboxes, dim=0).contiguous()
        labels = torch.cat(aug_labels).contiguous()
        if with_nms:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(
                bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = bboxes, labels

        return out_bboxes, out_labels

    # 粘贴的centernet
    # 图片翻转增强，是centernet原有的内容
    def aug_test(self, imgs, img_metas, rescale=True):
        """Augment testing of CenterNet. Aug test must have flipped image pair,
        and unlike CornerNet, it will perform an averaging operation on the
        feature map instead of detecting bbox.

        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.

        Note:
            ``imgs`` must including flipped image pairs.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img_inds = list(range(len(imgs)))
        # 确保输入图像对（imgs）包括了翻转的版本
        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
            'aug test must have flipped image pair')
        # 储存图像对的检测结果
        aug_results = []
        # 使用zip函数和步长2的切片操作，确保每次循环会处理一对图像
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            # 提取翻转后图像的翻转方向（水平或者垂直）
            flip_direction = img_metas[flip_ind][0]['flip_direction']
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            center_heatmap_preds, xywh_preds_coarse, xywh_preds_refine = self.bbox_head(x)
            assert len(center_heatmap_preds) == len(xywh_preds_coarse) == len(xywh_preds_refine) == 1

            # Feature map averaging
            center_heatmap_preds[0] = (
                center_heatmap_preds[0][0:1] +
                flip_tensor(center_heatmap_preds[0][1:2], flip_direction)) / 2
            xywh_preds_refine[0] = xywh_preds_refine[0][0:1]

            bbox_list = self.bbox_head.get_bboxes(
                center_heatmap_preds,
                xywh_preds_coarse,
                xywh_preds_refine,
                img_metas[ind],
                rescale=rescale,
                with_nms=False)
            aug_results.append(bbox_list)

        nms_cfg = self.bbox_head.test_cfg.get('nms_cfg', None)
        if nms_cfg is None:
            with_nms = False
        else:
            with_nms = True
        bbox_list = [self.merge_aug_results(aug_results, with_nms)]
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
