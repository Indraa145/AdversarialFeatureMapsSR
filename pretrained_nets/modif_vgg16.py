import torch
import torchvision
import numpy as np
import torch.nn as nn

class modif_vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())

        for bl in blocks:
            for p in bl:
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x1 = self.blocks[0](x)
        x2 = self.blocks[1](x1)
        x3 = self.blocks[2](x2)
        return x1, x2, x3

if __name__ == "__main__":
    new_model = modif_vgg16().cuda()
    x = np.random.randn(32, 3, 64, 64).astype(np.float32)  # Batch, ColorChannels, Height, Width
    x = torch.from_numpy(x).cuda()
    out1 = new_model(x)
    print(out1[0].shape)
    print(out1[1].shape)
    print(out1[2].shape)
    print("finished.")