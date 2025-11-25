import torch.nn as nn
import torch

class UNetAE(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        
        # ENCODER (guardamos features intermedias para skip connections)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),      # 128 -> 64
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),     # 64 -> 32
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),    # 32 -> 16
            nn.ReLU()
        )
        
        # BOTTLENECK
        self.enc_out = 128 * 16 * 16
        self.fc_enc = nn.Linear(self.enc_out, z_dim)
        self.fc_dec = nn.Linear(z_dim, self.enc_out)
        
        # DECODER (con skip connections - los canales se duplican por concatenación)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16 -> 32
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # 32 -> 64 (128 canales entrada por concat)
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1),    # 64 -> 128 (64 canales entrada por concat)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # ENCODER - guardamos features intermedias
        e1 = self.enc1(x)      # 64x64x32
        e2 = self.enc2(e1)     # 32x32x64
        e3 = self.enc3(e2)     # 16x16x128
        
        # BOTTLENECK
        e3_flat = e3.view(e3.size(0), -1)
        z = self.fc_enc(e3_flat)
        d3 = self.fc_dec(z)
        d3 = d3.view(-1, 128, 16, 16)
        
        # DECODER - con skip connections (concatenación)
        d3 = self.dec3(d3)                    # 32x32x64
        d3 = torch.cat([d3, e2], dim=1)       # Skip: concat con e2 -> 32x32x128
        
        d2 = self.dec2(d3)                    # 64x64x32
        d2 = torch.cat([d2, e1], dim=1)       # Skip: concat con e1 -> 64x64x64
        
        xhat = self.dec1(d2)                  # 128x128x3
        
        return xhat, z
