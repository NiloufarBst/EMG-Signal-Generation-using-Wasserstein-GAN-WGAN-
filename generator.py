import torch
import torch.nn as nn

latent_dim = 1000   #noise vector size
signal_length = 1000
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,signal_length),
            nn.Tanh()
        )

    def forward(self,z):
        return self.model(z)