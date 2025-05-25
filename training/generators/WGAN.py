import torch.nn as nn
class Generator(nn.Module):
    def __init__(self, input_dim, img_channels, features_g):
        super().__init__()
        self.net = nn.Sequential(
            self._block(input_dim, features_g * 8, 4, 1, 0),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)