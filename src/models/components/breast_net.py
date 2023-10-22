import torch
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s, efficientnet_v2_m, \
    EfficientNet_V2_M_Weights


class BreastNet(nn.Module):
    def __init__(self,
                 output_size: int = 6):
        super().__init__()
        weights = EfficientNet_V2_M_Weights.DEFAULT
        self.model = efficientnet_v2_m(weights)
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_size)  # Modify this layer for the specific number of classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == '__main__':
    breastnet = BreastNet().cuda()
    x = torch.randn(4, 3, 384, 384).cuda()
    result = breastnet(x)
    print(result)
