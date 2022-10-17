import torch
import torch.nn as nn


class IQA(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        ps = 25
        self.exposed_level = 0.5
        self.mean_pool = torch.nn.Sequential(torch.nn.ReflectionPad2d(ps // 2), torch.nn.AvgPool2d(ps, stride=1))

    def forward(self, images):
        eps = 1 / 255.0
        max_rgb = torch.max(images, dim=1, keepdim=True)[0]
        min_rgb = torch.min(images, dim=1, keepdim=True)[0]
        saturation = (max_rgb - min_rgb + eps) / (max_rgb + eps)

        mean_rgb = self.mean_pool(images).mean(dim=1, keepdim=True)
        exposedness = torch.abs(mean_rgb - self.exposed_level) + eps

        contrast = self.mean_pool(images * images).mean(dim=1, keepdim=True) - mean_rgb**2
        return torch.mean((saturation * contrast) / exposedness, dim=[1], keepdim=True)
