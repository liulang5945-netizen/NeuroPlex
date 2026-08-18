"""EnCodec 风格音频编解码器 — 把音频编码为离散 token id 序列。

与 TokenizerHub 的多模态接口对齐：
    codec.encode(audio: torch.Tensor) -> list[int]   # codebook 索引序列
    codec.decode(ids: list[int]) -> torch.Tensor      # 重建音频
    codec.vocab_size() -> int                          # 4096
    codec.eos_token_id() -> int                        # -1

架构：
    Encoder: 1D CNN 下采样（stride=128）→ [B, D, L/128] 连续特征
    Codebook: 最近邻量化，4096 entries × D dim
    Decoder: 1D CNN 上采样 → [B, 1, L] 重建音频

下采样率 128：1 秒音频（16kHz）→ 125 token。

注意：未训练的 codec 输出无意义，需训练后才有实际重建能力。
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroplex.config import MULTIMODAL_TOKENS


class _ResidualBlock1D(nn.Module):
    """1D 残差块。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(h + x)


class EnCodecEncoder(nn.Module):
    """音频编码器：1D CNN 下采样。"""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64, latent_dim: int = 128, stride: int = 128):
        super().__init__()
        # 下采样 stride=128：通过多层 stride 卷积实现
        # 128 = 2^7，用 7 层 stride=2 或组合
        # 简化：4 层 stride 卷积 (4*4*4*2=128)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 4, stride=4, padding=1),     # /4
            nn.ReLU(),
            _ResidualBlock1D(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 4, stride=4, padding=1),      # /16
            nn.ReLU(),
            _ResidualBlock1D(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),      # /32
            nn.ReLU(),
            _ResidualBlock1D(hidden_dim),
            nn.Conv1d(hidden_dim, latent_dim, 4, stride=4, padding=1),      # /128
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 1, L] → [B, D, L/128]"""
        return self.downsample(x)


class EnCodecDecoder(nn.Module):
    """音频解码器：1D CNN 上采样。"""

    def __init__(self, out_channels: int = 1, hidden_dim: int = 64, latent_dim: int = 128):
        super().__init__()
        self.upsample = nn.Sequential(
            _ResidualBlock1D(latent_dim),
            nn.ConvTranspose1d(latent_dim, hidden_dim, 4, stride=4, padding=1),    # x4
            nn.ReLU(),
            _ResidualBlock1D(hidden_dim),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=4, padding=1),    # x16
            nn.ReLU(),
            _ResidualBlock1D(hidden_dim),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),    # x32
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, out_channels, 4, stride=4, padding=1),  # x128
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, D, L/128] → [B, 1, L]"""
        return self.upsample(x)


class AudioQuantizer(nn.Module):
    """音频向量量化层（EMA codebook + dead code revival，防崩塌）。"""

    def __init__(self, num_embeddings: int = 4096, embedding_dim: int = 128,
                 commitment_cost: float = 0.25, ema_decay: float = 0.99,
                 dead_code_threshold: int = 100):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Kaiming 初始化（避免 uniform(-1/N,1/N) 初始值过小导致 codebook 崩塌）
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.kaiming_uniform_(self.codebook.weight)

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", self.codebook.weight.data.clone())
        self.register_buffer("usage_count", torch.zeros(num_embeddings, dtype=torch.long))
        self._ema_initialized = False

    def _init_ema_from_data(self, z_flat: torch.Tensor):
        """用第一批数据初始化 codebook（随机采样数据点作为初始码字）。"""
        N = z_flat.shape[0]
        n_init = min(N, self.num_embeddings)
        indices = torch.randperm(N, device=z_flat.device)[:n_init]
        self.codebook.weight.data[:n_init] = z_flat[indices]
        self.ema_w.data[:n_init] = z_flat[indices]
        self._ema_initialized = True

    def forward(self, z: torch.Tensor):
        """量化 [B, D, L] → (quantized [B, D, L], indices [B, L], loss)"""
        B, D, L = z.shape
        z_flat = z.permute(0, 2, 1).contiguous().view(-1, D)  # [B*L, D]

        if not self._ema_initialized and self.training:
            self._init_ema_from_data(z_flat)

        dist = (z_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_flat @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(dim=1))
        indices = dist.argmin(dim=1)  # [B*L]
        quantized_flat = self.codebook(indices)

        quantized = quantized_flat.view(B, L, D).permute(0, 2, 1).contiguous()
        indices_1d = indices.view(B, L)

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        # 只用 commitment loss（EMA 更新 codebook）
        commitment_loss = F.mse_loss(z, quantized.detach())
        loss = self.commitment_cost * commitment_loss

        # EMA 更新 + dead code revival
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.num_embeddings).float()
                cluster_size = one_hot.sum(dim=0)
                self.ema_cluster_size.data.mul_(self.ema_decay).add_(
                    cluster_size, alpha=1 - self.ema_decay)
                dw = one_hot.t() @ z_flat
                self.ema_w.data.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
                n = self.ema_cluster_size.sum()
                smoothed_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                self.codebook.weight.data.copy_(self.ema_w / smoothed_size.unsqueeze(1))
                self.usage_count.data += cluster_size.long()

                if self.usage_count.max() > self.dead_code_threshold:
                    dead_mask = self.usage_count < 1
                    n_dead = dead_mask.sum().item()
                    if n_dead > 0:
                        n_replace = min(n_dead, z_flat.shape[0])
                        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:n_replace]
                        rand_indices = torch.randperm(z_flat.shape[0], device=z_flat.device)[:n_replace]
                        self.codebook.weight.data[dead_indices] = z_flat[rand_indices].detach()
                        self.ema_w.data[dead_indices] = z_flat[rand_indices].detach()
                        self.usage_count.data[dead_indices] = 1

        return quantized_st, indices_1d, loss


class EnCodec(nn.Module):
    """完整 EnCodec 模型。"""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 128,
        num_embeddings: int = 4096,
        commitment_cost: float = 0.25,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.encoder = EnCodecEncoder(in_channels, hidden_dim, latent_dim)
        self.quantizer = AudioQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = EnCodecDecoder(in_channels, hidden_dim, latent_dim)
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        quantized, indices, vq_loss = self.quantizer(z)
        recon = self.decoder(quantized)
        # 强制对齐输出长度（ConvTranspose1d 可能差几个样本）
        if recon.shape[-1] != x.shape[-1]:
            target_len = x.shape[-1]
            if recon.shape[-1] > target_len:
                recon = recon[..., :target_len]
            else:
                recon = F.pad(recon, (0, target_len - recon.shape[-1]))
        return recon, indices, vq_loss

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices

    def decode_from_indices(self, indices: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        quantized = self.quantizer.codebook(indices)  # [B, L, D]
        quantized = quantized.permute(0, 2, 1).contiguous()  # [B, D, L]
        recon = self.decoder(quantized)
        if target_len is not None and recon.shape[-1] != target_len:
            if recon.shape[-1] > target_len:
                recon = recon[..., :target_len]
            else:
                recon = F.pad(recon, (0, target_len - recon.shape[-1]))
        return recon


class EnCodecAudioCodec:
    """音频编解码器 — 封装 EnCodec，满足 TokenizerHub 接口契约。

    Usage:
        codec = EnCodecAudioCodec()
        hub.register_modality("audio", codec)
        ids = hub.encode(audio_tensor, domain="general", modality="audio")
        recon = hub.decode(ids, domain="general", modality="audio")
    """

    def __init__(
        self,
        model: Optional[EnCodec] = None,
        sample_rate: int = 16000,
        device: Optional[torch.device] = None,
    ):
        self.model = model or EnCodec(sample_rate=sample_rate)
        self.model.eval()
        self.sample_rate = sample_rate
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self._codebook_size = MULTIMODAL_TOKENS["audio_codebook_size"]  # 4096

    def _preprocess(self, audio: torch.Tensor) -> torch.Tensor:
        """预处理：归一化到 [B, 1, L]。"""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # [L] → [1, 1, L]
        elif audio.dim() == 2:
            if audio.shape[0] > 1 and audio.shape[0] <= 2:
                # [channels, L] → [1, 1, L]（取单声道）
                audio = audio.mean(dim=0, keepdim=True).unsqueeze(0)
            else:
                # [B, L] → [B, 1, L]
                audio = audio.unsqueeze(1)
        audio = audio.float().to(self.device)
        # 归一化到 [-1, 1]
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max()
        return audio

    def encode(self, audio: torch.Tensor) -> List[int]:
        """音频 → codebook 索引序列。"""
        with torch.no_grad():
            x = self._preprocess(audio)
            indices = self.model.encode_to_indices(x)  # [B, L']
            return indices[0].tolist()

    def decode(self, ids: List[int]) -> torch.Tensor:
        """codebook 索引序列 → 重建音频。

        输出长度 = len(ids) * 128（每个 token 对应 128 个样本），
        与 encode 输入（若为 128 的倍数）长度一致。
        """
        with torch.no_grad():
            indices = torch.tensor([ids], dtype=torch.long, device=self.device)
            # 每个 token 对应 encoder stride=128 个样本
            target_len = len(ids) * 128
            recon = self.model.decode_from_indices(indices, target_len=target_len)  # [1, 1, L]
            return recon[0, 0].clamp(-1, 1)

    def vocab_size(self) -> int:
        return self._codebook_size

    def eos_token_id(self) -> int:
        return -1
