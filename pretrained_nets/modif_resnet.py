import torch
import torchvision
import numpy as np
import torch.nn as nn


class modif_resnet(nn.Module):
    def __init__(self, output_layer=None):
        super().__init__()
        self.pretrained = torchvision.models.resnet18(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers) - self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])

        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self, x):
        x = self.net(x)
        return x

if __name__ == "__main__":
    new_model = modif_resnet(output_layer = 'layer2').cuda() #inout 128 -> layer 2, input 64 -> layer 1; both output size are 16 (different channel num)
    x = np.random.randn(4, 3, 128, 128).astype(np.float32)  # Batch, ColorChannels, Height, Width
    x = torch.from_numpy(x).cuda()
    out1 = new_model(x)
    print(out1.shape)
    print("finished.")