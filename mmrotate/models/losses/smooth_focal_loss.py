# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss

from mmrotate.registry import MODELS
from functools import partial
import torch
from mmdet.models.losses.gfocal_loss import quality_focal_loss,quality_focal_loss_tensor_target, quality_focal_loss_with_prob
def smooth_focal_loss(pred,
                      target,
                      weight=None,
                      gamma=2.0,
                      alpha=0.25,
                      reduction='mean',
                      avg_factor=None):
    """Smooth Focal Loss proposed in Circular Smooth Label (CSL).

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The reduction method used to
            override the original reduction method of the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """

    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class SmoothFocalLoss(nn.Module):
    """Smooth Focal Loss. Implementation of `Circular Smooth Label (CSL).`__

    __ https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40

    Args:
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(SmoothFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_cls = self.loss_weight * smooth_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls


@MODELS.register_module()
class QualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
        activated (bool, optional): Whether the input is activated.
            If True, it means the input has been activated and can be
            treated as probabilities. Else, it should be treated as logits.
            Defaults to False.
    """

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False,
                 num_classes_1=6,
                 num_classes_2=2,):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated
        self.num_classes_1 = num_classes_1
        self.num_classes_2 = num_classes_2
        

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (Union(tuple([torch.Tensor]),Torch.Tensor)): The type is
                tuple, it should be included Target category label with
                shape (N,) and target quality label with shape (N,).The type
                is torch.Tensor, the target should be one-hot form with
                soft weights.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = quality_focal_loss_with_prob
            else:
                calculate_loss_func = quality_focal_loss
            if isinstance(target, torch.Tensor):
                # the target shape with (N,C) or (N,C,...), which means
                # the target is one-hot form with soft weights.
                calculate_loss_func = partial(
                    quality_focal_loss_tensor_target, activated=self.activated)
            loss = calculate_loss_func(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
            loss_cls = self.loss_weight * loss

        
        else:
            raise NotImplementedError
        return loss_cls
    