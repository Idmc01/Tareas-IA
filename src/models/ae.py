import torch.nn as nn
import torch

class ClassicAE(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64,128,4,2,1), nn.ReLU(),
            nn.Flatten()
        )

        self.enc_out = 128*16*16

        self.fc_enc = nn.Linear(self.enc_out, z_dim)
        self.fc_dec = nn.Linear(z_dim, self.enc_out)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128,16,16)),
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.fc_enc(self.encoder(x))
        xhat = self.decoder(self.fc_dec(z))
        return xhat, z
