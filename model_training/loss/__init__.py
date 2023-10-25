from typing import List

from torch.nn.modules.loss import _Loss, BCELoss
from torch import Tensor

from model_training.loss.matching import HungarianMatcher
from model_training.loss.util import get_iou, get_giou

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F






class DetrLoss(_Loss):
    def __init__(self, pos_rate: float) -> None:
        super().__init__()

        self.matcher = HungarianMatcher()
        self.pos_rate = pos_rate

        self.loss_class_weight = 1.0

        self.loss_boxes_weight = 1.0

    def forward(self, preds: Tensor, targets: List[Tensor]) -> Tensor:
        target_lens: List[int] = torch.tensor([len(target) for target in targets])
        max_target_len = max(target_lens)

        assert preds.shape[1] - max_target_len >= 0, 'Input has more CC regions than what can be predicted'

        targets_padded = torch.zeros(size=(preds.shape[0], preds.shape[1], 2), dtype=targets[0].dtype, device=targets[0].device)
        for batch_idx, batch_targets in enumerate(targets):
            batch_padding = [torch.zeros(size=(1, 2), device=preds.device) for _ in range(preds.shape[1] - len(batch_targets))]
            if len(batch_targets) > 0:
                batch_padding = [batch_targets] + batch_padding
            targets_padded[batch_idx] = torch.cat(batch_padding, dim=0)

        # 1.0 label exists prob for non padded targets 0.0 prob for padded targets
        targets_padded = torch.nn.functional.pad(targets_padded, (0, 1))
        for i, t_len in enumerate(target_lens):
            targets_padded[i, :t_len, -1] = 1.0




        matchings = self.matcher(outputs=preds, targets=targets_padded)

        targets_padded_reordered = preds.clone()
        targets_padded_reordered_padded_mask = torch.zeros(size=targets_padded_reordered.shape[:2], dtype=torch.bool, device=targets_padded_reordered.device)

        for batch_idx, matching in enumerate(matchings):
            targets_padded_reordered[batch_idx] = targets_padded[batch_idx, matching[1]]
            targets_padded_reordered_padded_mask[batch_idx] = (matching[1] < target_lens[batch_idx])

        loss = 0.0
        loss = self.loss_class_weight * self.loss_class(preds=preds, targets_padded_reordered=targets_padded_reordered, targets_padded_reordered_padded_mask=targets_padded_reordered_padded_mask)
        loss = loss + self.loss_boxes_weight * self.loss_boxes(preds=preds, targets_padded_reordered=targets_padded_reordered,targets_padded_reordered_padded_mask=targets_padded_reordered_padded_mask)

        return loss



    def loss_class(self, preds, targets_padded_reordered, targets_padded_reordered_padded_mask):
        pos_target_rate_per_batch = max(1e-3, targets_padded_reordered_padded_mask.sum() / targets_padded_reordered_padded_mask.numel())

        #weights = torch.full_like(preds[:, :, -1], pos_target_rate_per_batch)
        #weights[~targets_padded_reordered_padded_mask] = 1 - pos_target_rate_per_batch

        pred_probs = preds[:, :, -1]
        target_probs = targets_padded_reordered[:, :, -1]



        cost_class = - (
            (1) * target_probs * pred_probs.clamp(min=1e-8).log() +
                        1/10 * (1 - target_probs) * (1 - pred_probs).clamp(min=1e-8).log()
                        )


        return cost_class.mean()
        #return BCELoss(weight=weights)(preds[:, :, -1], targets_padded_reordered[:, :, -1])


    def loss_boxes(self, preds, targets_padded_reordered, targets_padded_reordered_padded_mask):

        non_padded_box_preds = preds[targets_padded_reordered_padded_mask]
        non_padded_box_targets = targets_padded_reordered[targets_padded_reordered_padded_mask]
        num_boxes = non_padded_box_preds.shape[0]


        pred_boxes = non_padded_box_preds[:, :2]
        target_boxes = non_padded_box_targets[:, :2]

        loss_dist = F.l1_loss(pred_boxes, target_boxes, reduction='none').sum() / num_boxes
        loss_giou = (num_boxes - get_giou(pred_boxes, target_boxes).sum()) / num_boxes

        loss_dist = loss_dist
        loss_giou = loss_giou


        return loss_dist + loss_giou
