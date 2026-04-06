import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------
# Time embedding (sinusoidal -> MLP)
# -------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.mlp(emb)

# -------------------------
# Residual Block with Time Embedding
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.res_skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.res_skip(x)

# -------------------------
# Self-Attention Block
# -------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, H * W).permute(0, 2, 1)
        k = self.k(h).reshape(B, C, H * W)
        v = self.v(h).reshape(B, C, H * W).permute(0, 2, 1)

        attn = torch.bmm(q, k) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)

# -------------------------
# Downsample and Upsample Blocks
# -------------------------
def downsample_conv(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

def upsample_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, 3, padding=1)
    )

# -------------------------
# Conditional U-Net
# -------------------------
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, time_emb_dim=256):
        super().__init__()

        # Time embedding
        self.time_emb = TimeEmbedding(time_emb_dim)

        # Encoder
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc2 = downsample_conv(base_channels, base_channels * 2)
        self.enc3 = downsample_conv(base_channels * 2, base_channels * 4)
        self.enc4 = downsample_conv(base_channels * 4, base_channels * 8)

        self.enc_blocks = nn.ModuleList([
            ResBlock(base_channels, base_channels, time_emb_dim),
            ResBlock(base_channels * 2, base_channels * 2, time_emb_dim),
            ResBlock(base_channels * 4, base_channels * 4, time_emb_dim),
            ResBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        ])

        self.enc_attn = nn.ModuleList([
            nn.Identity(), nn.Identity(),
            SelfAttentionBlock(base_channels * 4),
            SelfAttentionBlock(base_channels * 8)
        ])

        # Middle block
        self.middle_blocks = nn.ModuleList([
            ResBlock(base_channels * 8, base_channels * 8, time_emb_dim),
            ResBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        ])
        self.middle_attn = SelfAttentionBlock(base_channels * 8)

        # Decoder
        self.dec4_blocks = nn.ModuleList([
            ResBlock(base_channels * 16, base_channels * 8, time_emb_dim),
            ResBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        ])
        self.dec4_attn = SelfAttentionBlock(base_channels * 8)

        self.dec3_up = upsample_conv(base_channels * 8, base_channels * 4)
        self.dec3_blocks = nn.ModuleList([
            ResBlock(base_channels * 8, base_channels * 4, time_emb_dim),
            ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        ])
        self.dec3_attn = SelfAttentionBlock(base_channels * 4)

        self.dec2_up = upsample_conv(base_channels * 4, base_channels * 2)
        self.dec2_blocks = nn.ModuleList([
            ResBlock(base_channels * 4, base_channels * 2, time_emb_dim),
            ResBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        ])

        self.dec1_up = upsample_conv(base_channels * 2, base_channels)
        self.dec1_blocks = nn.ModuleList([
            ResBlock(base_channels * 2, base_channels, time_emb_dim),
            ResBlock(base_channels, base_channels, time_emb_dim)
        ])

        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)

    # -------------------------
    # Utility functions
    # -------------------------
    def apply_blocks(self, blocks, x, t_emb):
        for block in blocks:
            x = block(x, t_emb)
        return x

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x, t):
        t_emb = self.time_emb(t)

        # Encoder
        e1 = self.apply_blocks([self.enc_blocks[0]], self.enc1(x), t_emb)
        e2 = self.apply_blocks([self.enc_blocks[1]], self.enc2(e1), t_emb)
        e3 = self.apply_blocks([self.enc_blocks[2]], self.enc3(e2), t_emb)
        e4 = self.apply_blocks([self.enc_blocks[3]], self.enc4(e3), t_emb)

        e3, e4 = self.enc_attn[2](e3), self.enc_attn[3](e4)

        # Middle
        m = self.apply_blocks(self.middle_blocks, e4, t_emb)
        m = self.middle_attn(m)

        # Decoder
        d4 = torch.cat([m, e4], dim=1)  # no upsample here
        d4 = self.apply_blocks(self.dec4_blocks, d4, t_emb)
        d4 = self.dec4_attn(d4)

        d3 = self.dec3_up(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.apply_blocks(self.dec3_blocks, d3, t_emb)
        d3 = self.dec3_attn(d3)

        d2 = self.dec2_up(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.apply_blocks(self.dec2_blocks, d2, t_emb)

        d1 = self.dec1_up(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.apply_blocks(self.dec1_blocks, d1, t_emb)

        return self.final_conv(d1)