import torch

from model_training.loss import HungarianMatcher, DetrLoss

matcher = HungarianMatcher()


preds = torch.tensor([[0.5, 0.25, 0.8], [0.2, 0.1, 0.6], [0.1, 0.3, 0.9]]).unsqueeze(dim=0)
targets = torch.tensor([[0.5, 0.25, 1.0], [0.2, 0.1, 1.0], [0.0, 0.0, 0.0]]).unsqueeze(dim=0)



res = matcher(preds, targets)
print(res)


crit = DetrLoss(pos_rate=2/3)


targets = [targets[0, :2, :-1]]
res = crit(preds, targets)
print(res)