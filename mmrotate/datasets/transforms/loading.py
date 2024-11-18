# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import mmcv
from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import LoadAnnotations
import numpy as np

from mmrotate.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadAnnotationML(LoadAnnotations):

    def __init__(
            self,
            with_mask: bool = False,
            poly2mask: bool = True,
            box_type: str = 'hbox',
            # use for semseg
            reduce_zero_label: bool = False,
            ignore_index: int = 255,
            **kwargs) -> None:
        super(LoadAnnotationML, self).__init__(
            with_mask=with_mask,
            poly2mask=poly2mask,
            box_type=box_type,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)
    
    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels_1 = []
        gt_bboxes_labels_2 = []
        for instance in results.get('instances', []):
            gt_bboxes_labels_1.append(instance['bbox_label_1'])
            gt_bboxes_labels_2.append(instance['bbox_label_2'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels_1, dtype=np.int64)
        results['gt_bboxes_labels_2'] = np.array(
            gt_bboxes_labels_2, dtype=np.int64)
        
        
@TRANSFORMS.register_module()
class LoadPatchFromNDArray(BaseTransform):
    """Load a patch from the huge image w.r.t ``results['patch']``.

    Requaired Keys:

    - img
    - patch

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        pad_val (float or Sequence[float]): Values to be filled in padding
            areas. Defaults to 0.
    """

    def __init__(self,
                 pad_val: Union[float, Sequence[float]] = 0,
                 **kwargs) -> None:
        self.pad_val = pad_val

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with image array in ``results['img']``
                and patch position in ``results['patch']``.

        Returns:
            dict: The dict contains loaded patch and meta information.
        """
        image = results['img']
        img_h, img_w = image.shape[:2]

        patch_xmin, patch_ymin, patch_xmax, patch_ymax = results['patch']
        assert (patch_xmin < img_w) and (patch_xmax >= 0) and \
            (patch_ymin < img_h) and (patch_ymax >= 0)
        x1 = max(patch_xmin, 0)
        y1 = max(patch_ymin, 0)
        x2 = min(patch_xmax, img_w)
        y2 = min(patch_ymax, img_h)
        padding = (x1 - patch_xmin, y1 - patch_ymin, patch_xmax - x2,
                   patch_ymax - y2)

        patch = image[y1:y2, x1:x2]
        if any(padding):
            patch = mmcv.impad(patch, padding=padding, pad_val=self.pad_val)

        results['img_path'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape[:2]
        results['ori_shape'] = patch.shape[:2]
        return results
