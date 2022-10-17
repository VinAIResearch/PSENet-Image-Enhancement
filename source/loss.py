import torch


class TVLoss(torch.nn.Module):
    def forward(self, x):
        x = torch.log(x + 1e-3)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2)
        return torch.mean(h_tv) + torch.mean(w_tv)
