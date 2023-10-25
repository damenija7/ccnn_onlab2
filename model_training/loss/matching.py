import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from model_training.loss.util import get_iou_batch, get_giou_batch
from torch.nn import BCELoss


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs.shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs[:, :, -1].flatten()  # [batch_size * num_queries]
        out_bbox = outputs[:, :, :2].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and boxes
        tgt_prob = targets[:, :, -1].flatten()
        tgt_bbox = targets[:, :, :2].flatten(0, 1)
        tgt_empty_object_mask = tgt_prob < 0.5

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
#        cost_class = BCELoss(reduction='none')(out_prob, tgt_prob)
        #cost_class = - (((tgt_prob[None, :]) * (out_prob.clamp(min=1e-8).log()[:, None])) + (((1 - tgt_prob)[None, :]) * ((1 - out_prob).clamp(min=1e-8).log())[:, None]))
        cost_class = torch.zeros(size=(out_bbox.shape[0], tgt_bbox.shape[0]), dtype=tgt_prob.dtype, device=tgt_prob.device)
        cost_class[:, ~tgt_empty_object_mask] = -out_prob[:, None]
        cost_class[:, tgt_empty_object_mask] = 1.0
        # Account for class imbalance
        # TODO other than 10
        cost_class[:, tgt_empty_object_mask] /= 10

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox.to(torch.float), tgt_bbox.to(torch.float), p=1)
        # Compute the giou cost betwen boxes
        cost_giou = 1.0-get_giou_batch(out_bbox, tgt_bbox)

        # Dont count bbox and giou for empty objects
        cost_bbox[:, tgt_empty_object_mask] = 0.0
        cost_giou[:, tgt_empty_object_mask] = 0.0

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(target) for target in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64, device=outputs.device), torch.as_tensor(j, dtype=torch.int64, device=outputs.device)) for i, j in indices]