"""3D CNN 视频编解码器 — 把视频编码为离散 token id 序列。

与 TokenizerHub 的多模态接口对齐：
    codec.encode(video: torch.Tensor) -> list[int]   # codebook 索引序列
    codec.decode(ids: list[int]) -> torch.Tensor      # 重建视频
    codec.vocab_size() -> int                          # 8192（复用 image codebook）
    codec.eos_token_id() -> int                        # -1

架构：
    Encoder: 3D CNN 下采样（空间 4x + 时间 4x）→ [B, D, T/4, H/4, W/4]
    Codebook: EMA 量化 + 软分配熵正则，256 entries × 256 dim
    Decoder: 3D CNN 上采样 → [B, 3, T, H, W] 重建视频

输出 token 数 = (T/4) × (H/4) × (W/4)。
例如 16 帧 32×32 视频 → 4×8×8 = 256 token。

注意：未训练的 codec 输出无意义，需训练后才有实际重建能力。
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroplex.config import MULTIMODAL_TOKENS


class _ResidualBlock3D(nn.Module):
    """3D 残差块。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(h + x)


class VideoEncoder(nn.Module):
    """视频编码器：3D CNN 下采样（空间 4x + 时间 4x）。

    空间 4x + 时间 4x：16帧 32×32 → 4×8×8 = 256 token。
    kernel=4, stride=2, padding=1 精确互逆：L → L/2（无尺寸漂移）。
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, latent_dim: int = 256):
        super().__init__()
        self.downsample = nn.Sequential(
            # /2 时间 + /2 空间
            nn.Conv3d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            _ResidualBlock3D(hidden_dim),
            # /4 时间 + /4 空间（最终）
            nn.Conv3d(hidden_dim, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, T, H, W] → [B, D, T/4, H/4, W/4]"""
        return self.downsample(x)


class VideoDecoder(nn.Module):
    """视频解码器：3D CNN 上采样（空间 4x + 时间 4x）。

    kernel=4, stride=2, padding=1 精确互逆：L → 2L（与 encoder 严格对齐）。
    """

    def __init__(self, out_channels: int = 3, hidden_dim: int = 64, latent_dim: int = 256):
        super().__init__()
        self.upsample = nn.Sequential(
            _ResidualBlock3D(latent_dim),
            # x2 时间 + x2 空间
            nn.ConvTranspose3d(latent_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            _ResidualBlock3D(hidden_dim),
            # x2 时间 + x2 空间（最终）
            nn.ConvTranspose3d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, D, T/4, H/4, W/4] → [B, 3, T, H, W]"""
        return torch.sigmoid(self.upsample(x))


class VideoQuantizer(nn.Module):
    """视频向量量化层（EMA codebook + 持续死码重启，防崩塌）。

    复用音频 EnCodec 成功的 EMA 策略，但修复死码重启 bug：
    用 EMA 跟踪近期使用率（而非累计计数），持续重启死码。
    """

    def __init__(self, num_embeddings: int = 8192, embedding_dim: int = 256,
                 commitment_cost: float = 0.25, ema_decay: float = 0.99,
                 dead_threshold: float = 1e-3, diversity_cost: float = 0.1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.dead_threshold = dead_threshold
        self.diversity_cost = diversity_cost  # z 方差正则权重

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.kaiming_uniform_(self.codebook.weight)

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", self.codebook.weight.data.clone())
        # EMA 近期使用率（持续识别死码，非累计）
        self.register_buffer("ema_usage", torch.zeros(num_embeddings))
        self._ema_initialized = False

    def _init_ema_from_data(self, z_flat: torch.Tensor):
        """用第一批数据初始化 codebook。"""
        N = z_flat.shape[0]
        n_init = min(N, self.num_embeddings)
        indices = torch.randperm(N, device=z_flat.device)[:n_init]
        self.codebook.weight.data[:n_init] = z_flat[indices]
        self.ema_w.data[:n_init] = z_flat[indices]
        self._ema_initialized = True

    def forward(self, z: torch.Tensor):
        """量化 [B, D, T', H', W'] → (quantized, indices [B, T', H', W'], loss)"""
        B, D, T, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, D)  # [B*T*H*W, D]
        N = z_flat.shape[0]

        if not self._ema_initialized and self.training:
            self._init_ema_from_data(z_flat)

        dist = (z_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_flat @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(dim=1))
        indices = dist.argmin(dim=1)
        quantized_flat = self.codebook(indices)

        quantized = quantized_flat.view(B, T, H, W, D).permute(0, 4, 1, 2, 3).contiguous()
        indices_3d = indices.view(B, T, H, W)

        # STE
        quantized_st = z + (quantized - z).detach()

        # commitment loss（EMA 更新 codebook，不需要 codebook_loss）
        commitment_loss = F.mse_loss(z, quantized.detach())
        # 软分配熵正则（有梯度，直接鼓励码字均匀使用）
        # 用 softmax(dist) 作为软分配，最大化其熵 ↔ 最小化 -H
        soft_prob = F.softmax(-dist, dim=-1)  # [N, num_embeddings]
        avg_prob = soft_prob.mean(dim=0)  # [num_embeddings]
        entropy = -(avg_prob * (avg_prob + 1e-8).log()).sum()
        loss = self.commitment_cost * commitment_loss - self.diversity_cost * entropy

        # EMA 更新 + 持续死码重启
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.num_embeddings).float()
                cluster_size = one_hot.sum(dim=0)  # [num_embeddings]

                # EMA codebook 更新
                self.ema_cluster_size.data.mul_(self.ema_decay).add_(
                    cluster_size, alpha=1 - self.ema_decay)
                dw = one_hot.t() @ z_flat
                self.ema_w.data.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
                n = self.ema_cluster_size.sum()
                smoothed_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                self.codebook.weight.data.copy_(self.ema_w / smoothed_size.unsqueeze(1))

                # EMA 近期使用率（归一化概率）
                batch_prob = cluster_size / N
                self.ema_usage.data.mul_(self.ema_decay).add_(batch_prob, alpha=1 - self.ema_decay)

                # 持续死码重启：EMA 使用率低于阈值 → 用 batch 随机点替换
                dead_mask = self.ema_usage < self.dead_threshold
                n_dead = dead_mask.sum().item()
                if n_dead > 0 and N > 0:
                    n_replace = min(n_dead, N)
                    dead_indices = dead_mask.nonzero(as_tuple=True)[0][:n_replace]
                    rand_indices = torch.randperm(N, device=z_flat.device)[:n_replace]
                    new_codes = z_flat[rand_indices].detach()
                    new_codes = new_codes + torch.randn_like(new_codes) * 0.1
                    self.codebook.weight.data[dead_indices] = new_codes
                    self.ema_w.data[dead_indices] = new_codes
                    # 关键：同步重置 ema_cluster_size，避免 EMA 归一化放大
                    self.ema_cluster_size.data[dead_indices] = 1.0
                    self.ema_usage.data[dead_indices] = self.dead_threshold * 2

        return quantized_st, indices_3d, loss


class VideoVQVAE(nn.Module):
    """完整视频 VQ-VAE 模型。"""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        latent_dim: int = 256,
        num_embeddings: int = 8192,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.encoder = VideoEncoder(in_channels, hidden_dim, latent_dim)
        self.quantizer = VideoQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = VideoDecoder(in_channels, hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        quantized, indices, vq_loss = self.quantizer(z)
        recon = self.decoder(quantized)
        # 强制对齐输出形状（3D ConvTranspose 可能差几个样本）
        if recon.shape != x.shape:
            target = x.shape
            # 裁剪到目标形状（时间、空间都可能略大）
            recon = recon[:, :, :target[2], :target[3], :target[4]]
            # 若裁剪后仍不足（罕见），补零
            if recon.shape != target:
                pad = [0, target[4] - recon.shape[4],
                       0, target[3] - recon.shape[3],
                       0, target[2] - recon.shape[2]]
                recon = F.pad(recon, pad)
        return recon, indices, vq_loss

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices

    def decode_from_indices(self, indices: torch.Tensor, target_shape: Optional[tuple] = None) -> torch.Tensor:
        B, T, H, W = indices.shape
        quantized = self.quantizer.codebook(indices)  # [B, T, H, W, D]
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()  # [B, D, T, H, W]
        recon = self.decoder(quantized)
        if target_shape is not None:
            # target_shape = (T, H, W)
            t, h, w = target_shape
            recon = recon[:, :, :t, :h, :w]
            if recon.shape[2] < t or recon.shape[3] < h or recon.shape[4] < w:
                pad = [0, w - recon.shape[4],
                       0, h - recon.shape[3],
                       0, t - recon.shape[2]]
                recon = F.pad(recon, pad)
        return recon


class VideoCodec:
    """视频编解码器 — 封装 VideoVQVAE，满足 TokenizerHub 接口契约。

    Usage:
        codec = VideoCodec()
        hub.register_modality("video", codec)
        ids = hub.encode(video_tensor, domain="general", modality="video")
        recon = hub.decode(ids, domain="general", modality="video")
    """

    def __init__(
        self,
        model: Optional[VideoVQVAE] = None,
        frame_size: int = 224,
        num_frames: int = 16,
        device: Optional[torch.device] = None,
    ):
        self.model = model or VideoVQVAE()
        self.model.eval()
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self._codebook_size = MULTIMODAL_TOKENS["video_codebook_size"]

    def _preprocess(self, video: torch.Tensor) -> torch.Tensor:
        """预处理：归一化到 [B, 3, T, H, W]。"""
        if video.dim() == 4:
            # 判断输入格式：
            # [T, C, H, W] — shape[1] in (1,3)，最常见
            # [C, T, H, W] — shape[0] in (1,3)
            # [T, H, W, C] — shape[-1] in (1,3)
            if video.shape[1] in (1, 3):
                # [T, C, H, W] → [1, C, T, H, W]
                video = video.permute(1, 0, 2, 3).unsqueeze(0)
            elif video.shape[0] in (1, 3):
                # [C, T, H, W] → [1, C, T, H, W]
                video = video.unsqueeze(0)
            elif video.shape[-1] in (1, 3):
                # [T, H, W, C] → [1, C, T, H, W]
                video = video.permute(3, 0, 1, 2).unsqueeze(0)
            else:
                raise ValueError(f"无法推断 4D 视频输入格式: {video.shape}")
        elif video.dim() == 5:
            pass  # [B, 3, T, H, W]

        video = video.float().to(self.device)
        if video.max() > 1.0:
            video = video / 255.0

        # Resize 空间维度到 frame_size
        B, C, T, H, W = video.shape
        if H != self.frame_size or W != self.frame_size:
            video = video.view(B * T, C, H, W)
            video = F.interpolate(video, size=(self.frame_size, self.frame_size), mode="bilinear")
            video = video.view(B, C, T, self.frame_size, self.frame_size)

        # 裁剪/补齐时间维度到 num_frames
        if T > self.num_frames:
            video = video[:, :, :self.num_frames]
        elif T < self.num_frames:
            pad = torch.zeros(B, C, self.num_frames - T, self.frame_size, self.frame_size,
                              device=self.device)
            video = torch.cat([video, pad], dim=2)

        return video

    def encode(self, video: torch.Tensor) -> List[int]:
        """视频 → codebook 索引序列。"""
        with torch.no_grad():
            x = self._preprocess(video)
            indices = self.model.encode_to_indices(x)  # [B, T', H', W']
            return indices[0].flatten().tolist()

    def decode(self, ids: List[int]) -> torch.Tensor:
        """codebook 索引序列 → 重建视频。"""
        with torch.no_grad():
            n = len(ids)
            # 推断 3D 尺寸：T' = num_frames/4, H' = W' = frame_size/4（空间 4x 下采样）
            t = self.num_frames // 4
            s = self.frame_size // 4
            total = t * s * s
            if len(ids) < total:
                ids = ids + [0] * (total - len(ids))
            elif len(ids) > total:
                ids = ids[:total]
            indices = torch.tensor(ids, dtype=torch.long, device=self.device).view(1, t, s, s)
            recon = self.model.decode_from_indices(
                indices, target_shape=(self.num_frames, self.frame_size, self.frame_size)
            )  # [1, 3, T, H, W]
            return recon[0].clamp(0, 1)

    def vocab_size(self) -> int:
        return self._codebook_size

    def eos_token_id(self) -> int:
        return -1
