import torch
from torch import nn
import torch.nn.functional as F


class WeightedCriterion(nn.Module):
    def __init__(self, criteria):
        super().__init__()
        self.criteria = criteria

    def forward(self, x):
        # loss_dict = {}
        loss = 0
        for criterion in self.criteria:
            temp_loss = criterion["criterion"](x[criterion["key"]]) * criterion["weight"]
            print(f"{criterion['key']} weighted loss: {temp_loss}")
            print(f"{criterion['key']} weight: {criterion['weight']}")
            loss += temp_loss
            # loss_dict[criterion["key"]] = temp_loss
        return loss

