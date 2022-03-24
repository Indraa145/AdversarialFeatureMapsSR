import torch.nn as nn
import torch

from generator_l2h import BasicBlock
import numpy as np
import cv2

class High2Low(nn.Module):
    def __init__(self):
        super(High2Low, self).__init__()
        blocks = [96, 96, 128, 128, 256, 256, 512, 512, 128, 128, 32, 32]

        self.noise_fc = nn.Linear(64, 4096)
        self.in_layer = nn.Conv2d(4, blocks[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.out_layer = nn.Sequential(
            nn.Conv2d(blocks[-1], 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        downs = []
        in_feat = blocks[0]
        for i in range(8): # downsample layers
            b_down = not i % 2
            downs.append(BasicBlock(in_feat, blocks[i], downsample=b_down))
            in_feat = blocks[i]

        ups = []
        for i in range(2):
            ups.append(nn.PixelShuffle(2))
            ups.append(BasicBlock(blocks[8+i*2], blocks[8+i*2]))
            ups.append(BasicBlock(blocks[9+i*2], blocks[9+i*2]))

        self.down_layers = nn.Sequential(*downs)
        self.up_layers = nn.Sequential(*ups)

    def forward(self, x, z):
        noises = self.noise_fc(z)
        noises = noises.view(-1, 1, 64, 64)
        out = torch.cat((x, noises), 1)
        out = self.in_layer(out)
        out = self.down_layers(out)
        out = self.up_layers(out)
        out = self.out_layer(out)
        return out

def high2low_test():
    net = High2Low().cuda()
    Z = np.random.randn(1, 1, 64).astype(np.float32)
    X = np.random.randn(1, 3, 64, 64).astype(np.float32)
    Z = torch.from_numpy(Z).cuda()
    X = torch.from_numpy(X).cuda()
    Y = net(X, Z)
    print(Y.shape)
    Xim = X.cpu().numpy().squeeze().transpose(1, 2, 0)
    Yim = Y.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    Xim = (Xim - Xim.min()) / (Xim.max() - Xim.min())
    Yim = (Yim - Yim.min()) / (Yim.max() - Yim.min())
    cv2.imshow("X", Xim)
    cv2.imshow("Y", Yim)
    cv2.waitKey()

if __name__ == "__main__":
    high2low_test()
    print("finished.")