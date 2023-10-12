import torch


CENTER_IDX = 0
WIDTH_IDX = 1

class YOLO(torch.nn.Module):
    def __init__(self, num_grids: int = 16, orig_size: int = 1024):
        super().__init__()

        self.grid_size = orig_size / num_grids
        self.num_grids = num_grids




    # call once for every batch element
    def _iou(self, ground_truth_box: torch.Tensor, pred_box: torch.Tensor):
        #ground_truth_box shape: (4)
        # pred_box_shape: (num_grids, 4)

        ground_truth_left, ground_truth_right = ground_truth_box[CENTER_IDX] - ground_truth_box[WIDTH_IDX] / 2, ground_truth_box[CENTER_IDX] + ground_truth_box[WIDTH_IDX] / 2

        pred_box_lefts, pred_box_rights = pred_box[:, CENTER_IDX] - pred_box[:, WIDTH_IDX] / 2, pred_box[:, CENTER_IDX] + pred_box[:, WIDTH_IDX] / 2

        pass

        return unions_per_grid

