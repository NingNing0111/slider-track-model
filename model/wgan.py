"""Conditional WGAN-GP：Generator 与 Discriminator（v2: 2D 输出, 无 dt）。"""
import torch
import torch.nn as nn

LATENT_DIM = 64
COND_DIM = 1
MAX_LEN = 128       # 重采样后固定长度
FEAT = 2             # dx, dy（去掉 dt，用固定时间间隔）


class Generator(nn.Module):
    """输入: z (batch, latent_dim), c (batch, 1); 输出: (batch, max_len, 2)."""

    def __init__(
        self,
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        max_len=MAX_LEN,
        hidden=128,
        step_noise_dim=32,
        step_noise_alpha=0.4,
    ):
        super().__init__()
        self.max_len = max_len
        self.step_noise_dim = int(step_noise_dim)
        self.step_noise_alpha = float(step_noise_alpha)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
        )
        self.pos_embedding = nn.Embedding(max_len, hidden)
        # ONNX 友好：位置索引作为常量 buffer，避免 forward 内创建 arange
        self.register_buffer("pos_idx", torch.arange(max_len, dtype=torch.long), persistent=False)
        self.step_noise_proj = nn.Linear(self.step_noise_dim, hidden) if self.step_noise_dim > 0 else None
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, FEAT),
        )

    def forward(self, z, c, step_noise=None):
        batch_size = z.size(0)
        h = self.proj(torch.cat([z, c], dim=1))                            # (B, H)
        h = h.unsqueeze(1).expand(-1, self.max_len, -1)                    # (B, L, H)
        pos_emb = self.pos_embedding(self.pos_idx.unsqueeze(0).expand(batch_size, -1))  # (B, L, H)
        lstm_in = h + pos_emb
        # 逐步噪声：让生成器在局部也有随机性，避免只输出过于平滑的“均值轨迹”
        if self.step_noise_proj is not None and self.step_noise_alpha > 0:
            if step_noise is None:
                step_noise = torch.randn(
                    batch_size,
                    self.max_len,
                    self.step_noise_dim,
                    device=z.device,
                    dtype=z.dtype,
                )
            lstm_in = lstm_in + self.step_noise_alpha * self.step_noise_proj(step_noise)
        out, _ = self.lstm(lstm_in)
        return self.out(out)                                               # (B, L, 2)


class Discriminator(nn.Module):
    """输入: seq (batch, max_len, 2), c (batch, 1); 输出: (batch, 1) score。"""

    def __init__(self, max_len=MAX_LEN, cond_dim=COND_DIM, channels=(32, 64, 128)):
        super().__init__()
        layers = []
        in_c = FEAT + cond_dim   # dx, dy + condition
        for i, ch in enumerate(channels):
            # 第一层不下采样，更敏感高频/局部细节；后续再做 stride=2 下采样
            stride = 1 if i == 0 else 2
            layers += [
                nn.Conv1d(in_c, ch, 5, padding=2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(ch, ch, 3, stride=stride, padding=1),
                nn.LeakyReLU(0.2),
            ]
            in_c = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_c, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, seq, c):
        c_expand = c.unsqueeze(1).expand(-1, seq.size(1), -1)
        x = torch.cat([seq, c_expand], dim=2)
        x = x.transpose(1, 2)   # (B, C, L) for Conv1d
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
