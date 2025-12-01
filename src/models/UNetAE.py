import torch
import torch.nn as nn


class UNetAE(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU()
        )

        # Bottleneck
        self.enc_out = 128 * 16 * 16
        self.fc_enc = nn.Linear(self.enc_out, z_dim)
        self.fc_dec = nn.Linear(z_dim, self.enc_out)

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, 2, 1),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        e3_flat = e3.view(e3.size(0), -1)
        z = self.fc_enc(e3_flat)
        d3 = self.fc_dec(z).view(-1, 128, 16, 16)

        # Decoder + skips
        d3 = self.dec3(d3)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        xhat = self.dec1(d2)
        return xhat, z