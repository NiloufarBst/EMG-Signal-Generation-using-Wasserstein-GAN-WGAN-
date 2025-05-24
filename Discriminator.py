import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1000,128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128,64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64,1)
        )

    def forward(self, x):
        return self.model(x)
