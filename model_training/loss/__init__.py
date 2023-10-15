from typing import List

from torch.nn.modules.loss import _Loss, BCELoss
from torch import Tensor

from model_training.loss.matching import HungarianMatcher
from model_training.loss.util import get_iou

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

    def forward(self, preds: Tensor, targets: List[Tensor]) -> Tensor:
        target_lens: List[int] = [len(target) for target in targets]
        targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

        targets_padded_with_empty_targets = torch.cat((targets_padded,torch.zeros(size=(preds.shape[0], preds.shape[1] - targets_padded.shape[1], targets_padded.shape[-1]), dtype=targets_padded.dtype, device=targets_padded.device)),
                                                        dim=1)
        targets_padded_with_empty_targets = torch.nn.functional.pad(targets_padded_with_empty_targets, (0, 1))


        for i, t_len in enumerate(target_lens):
            targets_padded_with_empty_targets[i, :t_len, -1] = 1.0




        matchings = self.matcher(outputs=preds, targets=targets_padded_with_empty_targets)
        #matchings = torch.stack([matching[1] for matching in matchings])

        preds_reordered = preds.clone()

        for batch_idx, matching in enumerate(matchings):
            preds_reordered[batch_idx] = preds[batch_idx, matching[1]]

        loss = self.loss_labels(preds=preds, targets_padded_with_empty_targets=targets_padded_with_empty_targets, target_lens=target_lens)
        loss = loss + self.loss_boxes(preds=preds, targets=targets, target_lens=target_lens)

        return loss




    def loss_labels(self, preds, targets_padded_with_empty_targets, target_lens):
        weights = torch.full_like(preds[:, :, -1], self.pos_rate)

        for batch_idx, target_len in enumerate(target_lens):
            weights[batch_idx, :target_len] = 1 - self.pos_rate

        return BCELoss(weight=weights)(preds[:, :, -1], targets_padded_with_empty_targets[:, :, -1])


    def loss_boxes(self, preds, targets, target_lens):
        loss_dist = 0.0
        loss_iou = 0.0

        num_boxes = 0

        for batch_idx, target_len in enumerate(target_lens):
            pred_boxes = preds[batch_idx, :target_len, :2]
            target_boxes = targets[batch_idx][:target_len, :2]

            loss_dist = loss_dist + F.l1_loss(pred_boxes,
                                    target_boxes, reduction='none').sum()
            num_boxes += target_len

            loss_iou = loss_iou + get_iou(pred_boxes, target_boxes).sum()

        loss_dist = loss_dist / num_boxes
        loss_iou = loss_iou / num_boxes


        return loss_dist + loss_iou
