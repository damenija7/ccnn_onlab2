import torch

def get_iou_batch(boxes_1, boxes_2):
    return _get_iou_batch(boxes_1, boxes_2)[0]

def _get_iou_batch(boxes_1: torch.Tensor, boxes_2: torch.Tensor):
    # boxes_1: (N, 2), boxes_2: (M, 2)

    # boxes_i_l,_r = (N/M)
    boxes_1_left, boxes_1_right = boxes_1[:, 0] - boxes_1[:, 1]/2, boxes_1[:, 0] + boxes_1[:, 1]/2
    boxes_2_left, boxes_2_right = boxes_2[:, 0] - boxes_2[:, 1] / 2, boxes_2[:, 0] + boxes_2[:, 1] / 2

    #iou = torch.zeros(size=(boxes_1.shape[0], boxes_2.shape[0]), device=boxes_1.device)

    iou = torch.min(boxes_1_right[:, None], boxes_2_right[None, :]) - torch.max(boxes_1_left[:, None], boxes_2_left[None, :])
    # iou = intersect / union
    union = (((boxes_1_right - boxes_1_left)[:, None] + (boxes_2_right - boxes_2_left)[None, :]) - iou)
    iou = iou / union

    # for i in range(boxes_1.shape[0]):
    #     for j in range(boxes_2.shape[0]):
    #         single_intersection = min(boxes_1_right[i], boxes_2_right[j]) - max(boxes_1_left[i], boxes_2_left[j])
    #         single_union = (boxes_1_right[i] - boxes_1_left[i]) + (boxes_2_right[j] - boxes_2_left[j]) - single_intersection
    #
    #         iou[i, j] = single_intersection / single_union

    iou = iou.clamp(min=0.0)

    iou = torch.nan_to_num(iou)

    assert not torch.isnan(iou).any() and not torch.isnan(union).any()

    return iou, union

def get_giou_batch(boxes_1, boxes_2):
    # https://giou.stanford.edu/

    iou, union = _get_iou_batch(boxes_1, boxes_2)

    boxes_1_left, boxes_1_right = boxes_1[:, 0] - boxes_1[:, 1] / 2, boxes_1[:, 0] + boxes_1[:, 1] / 2
    boxes_2_left, boxes_2_right = boxes_2[:, 0] - boxes_2[:, 1] / 2, boxes_2[:, 0] + boxes_2[:, 1] / 2

    # smallest convex hull enclosing both
    hull = (torch.max(boxes_1_right[:, None], boxes_2_right[None, :]) - torch.min(boxes_1_left[:, None],
                                                                                boxes_2_left[None, :])).clamp(min=0.0)

    res = iou - (hull - union) / hull

    assert not torch.isnan(res).any()

    return res



def get_giou(boxes_1, boxes_2 ):
    iou, union = _get_iou(boxes_1, boxes_2)

    boxes_1_left, boxes_1_right = boxes_1[:, 0] - boxes_1[:, 1] / 2, boxes_1[:, 0] + boxes_1[:, 1] / 2
    boxes_2_left, boxes_2_right = boxes_2[:, 0] - boxes_2[:, 1] / 2, boxes_2[:, 0] + boxes_2[:, 1] / 2

    hull = torch.max(boxes_1_right, boxes_2_right) - torch.min(boxes_1_left, boxes_2_left).clamp(min=0.0)

    res = iou - (hull - union) / hull

    return res



def _get_iou(boxes_1: torch.Tensor, boxes_2: torch.Tensor):
    boxes_1_left, boxes_1_right = boxes_1[:, 0] - boxes_1[:, 1]/2, boxes_1[:, 0] + boxes_1[:, 1]/2
    boxes_2_left, boxes_2_right = boxes_2[:, 0] - boxes_2[:, 1] / 2, boxes_2[:, 0] + boxes_2[:, 1] / 2

    iou = torch.min(boxes_1_right, boxes_2_right) - torch.max(boxes_1_left, boxes_2_left)
    union = ((boxes_1_right - boxes_1_left) + (boxes_2_right - boxes_2_left) - iou)
    iou = iou / union

    return iou.clamp(min=0.0), union

def get_iou(boxes_1, boxes_2):
    return _get_iou(boxes_1, boxes_2)[0]

