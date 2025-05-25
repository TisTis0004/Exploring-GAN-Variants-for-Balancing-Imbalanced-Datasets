import torch.nn as nn
class Generator(nn.Module):
    def __init__(self,input_dim, out_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, out_dim),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.gen(x) 