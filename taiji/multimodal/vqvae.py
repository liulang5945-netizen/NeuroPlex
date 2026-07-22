"""VQ-VAE 图像编解码器 — 把图像编码为离散 token id 序列。

与 TokenizerHub 的多模态接口对齐：
    codec.encode(image: torch.Tensor) -> list[int]   # codebook 索引序列
    codec.decode(ids: list[int]) -> torch.Tensor      # 重建图像
    codec.vocab_size() -> int                          # 8192
    codec.eos_token_id() -> int                        # -1（图像无 EOS）

架构：
    Encoder: CNN 下采样 16x → [B, D, H/16, W/16] 连续特征图
    Codebook: 最近邻量化，8192 entries × D dim
    Decoder: CNN 上采样 16x → [B, 3, H, W] 重建图像

输出 token 数 = (H/16) × (W/16)，例如 224×224 输入 → 14×14 = 196 token。

注意：未训练的 VQ-VAE 输出无意义，需训练后才有实际重建能力。
本实现聚焦架构骨架，训练流程后续补充。
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.config import MULTIMODAL_TOKENS


class _ResidualBlock(nn.Module):
    """残差块（VQ-VAE encoder/decoder 基础单元）。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(h + x)


class VQVAEEncoder(nn.Module):
    """VQ-VAE 编码器：图像 → 连续特征图 [B, D, H/downsample, W/downsample]。

    Args:
        downsample: 下采样倍数（4, 8, 16）。默认 8（平衡 token 数和重建质量）。
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 128, latent_dim: int = 256,
                 downsample: int = 8):
        super().__init__()
        self.downsample_factor = downsample

        if downsample == 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),  # /2
                nn.ReLU(),
                _ResidualBlock(hidden_dim),
                nn.Conv2d(hidden_dim, latent_dim, 4, stride=2, padding=1),   # /4
                nn.ReLU(),
            )
        elif downsample == 8:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),  # /2
                nn.ReLU(),
                _ResidualBlock(hidden_dim),
                nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),   # /4
                nn.ReLU(),
                _ResidualBlock(hidden_dim),
                nn.Conv2d(hidden_dim, latent_dim, 4, stride=2, padding=1),   # /8
                nn.ReLU(),
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),  # /2
                nn.ReLU(),
                _ResidualBlock(hidden_dim),
                nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),   # /4
                nn.ReLU(),
                _ResidualBlock(hidden_dim),
                nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),   # /8
                nn.ReLU(),
                _ResidualBlock(hidden_dim),
                nn.Conv2d(hidden_dim, latent_dim, 4, stride=2, padding=1),   # /16
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)


class VQVAEDecoder(nn.Module):
    """VQ-VAE 解码器：量化特征图 → 重建图像。

    Args:
        downsample: 上采样倍数（与 encoder 对应）。
    """

    def __init__(self, out_channels: int = 3, hidden_dim: int = 128, latent_dim: int = 256,
                 downsample: int = 8):
        super().__init__()
        self.downsample_factor = downsample

        if downsample == 4:
            self.upsample = nn.Sequential(
                _ResidualBlock(latent_dim),
                nn.ConvTranspose2d(latent_dim, hidden_dim, 4, stride=2, padding=1),  # x2
                nn.ReLU(),
                _ResidualBlock(hidden_dim),
                nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1),  # x4
            )
        elif downsample == 8:
            self.upsample = nn.Sequential(
                _ResidualBlock(latent_dim),
                nn.ConvTranspose2d(latent_dim, hidden_dim, 4, stride=2, padding=1),  # x2
                nn.ReLU(),
                _ResidualBlock(hidden_dim),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # x4
                nn.ReLU(),
                _ResidualBlock(hidden_dim),
                nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1),  # x8
            )
        else:
            self.upsample = nn.Sequential(
                _ResidualBlock(latent_dim),
                nn.ConvTranspose2d(latent_dim, hidden_dim, 4, stride=2, padding=1),  # x2
                nn.ReLU(),
                _ResidualBlock(hidden_dim),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # x4
                nn.ReLU(),
                _ResidualBlock(hidden_dim),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # x8
                nn.ReLU(),
                _ResidualBlock(hidden_dim),
                nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1),  # x16
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class VectorQuantizer(nn.Module):
    """向量量化层：最近邻查找 codebook 索引。

    使用 straight-through estimator (STE) 梯度传递。
    EMA codebook 更新 + dead code revival 防止 codebook 崩塌。
    """

    def __init__(self, num_embeddings: int = 8192, embedding_dim: int = 256,
                 commitment_cost: float = 0.25, ema_decay: float = 0.99,
                 dead_code_threshold: int = 100):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Codebook: [num_embeddings, embedding_dim]
        # 用 kaiming 初始化代替 uniform(-1/N, 1/N)，避免初始值过小
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.kaiming_uniform_(self.codebook.weight)

        # EMA 统计
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", self.codebook.weight.data.clone())
        self.register_buffer("usage_count", torch.zeros(num_embeddings, dtype=torch.long))
        self._ema_initialized = False

    def _init_ema_from_data(self, z_flat: torch.Tensor):
        """用第一批数据初始化 codebook（k-means++ 风格：随机采样数据点作为初始码字）。"""
        N = z_flat.shape[0]
        n_init = min(N, self.num_embeddings)
        indices = torch.randperm(N, device=z_flat.device)[:n_init]
        self.codebook.weight.data[:n_init] = z_flat[indices]
        self.ema_w.data[:n_init] = z_flat[indices]
        self._ema_initialized = True

    def forward(self, z: torch.Tensor):
        """量化连续特征。

        Args:
            z: [B, D, H, W] 连续特征

        Returns:
            quantized: [B, D, H, W] 量化后特征（STE 梯度传递）
            indices: [B, H, W] codebook 索引
            loss: commitment + codebook loss
        """
        B, D, H, W = z.shape
        # 展平为 [B*H*W, D]
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # 首次 forward 时用数据初始化 codebook
        if not self._ema_initialized and self.training:
            self._init_ema_from_data(z_flat)

        # 计算与 codebook 的距离（||z - e||^2 = ||z||^2 - 2 z·e + ||e||^2）
        dist = (z_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_flat @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(dim=1))

        # 最近邻索引
        indices = dist.argmin(dim=1)  # [B*H*W]
        quantized_flat = self.codebook(indices)  # [B*H*W, D]

        # 还原形状
        quantized = quantized_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        indices_2d = indices.view(B, H, W)

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        # Loss: 只用 commitment loss（EMA 更新 codebook，不需要 codebook_loss）
        commitment_loss = F.mse_loss(z, quantized.detach())
        loss = self.commitment_cost * commitment_loss

        # EMA 更新 codebook
        if self.training:
            with torch.no_grad():
                # 统计每个码字的使用次数
                one_hot = F.one_hot(indices, self.num_embeddings).float()  # [N, num_emb]
                cluster_size = one_hot.sum(dim=0)  # [num_emb]
                # EMA 更新 cluster size
                self.ema_cluster_size.data.mul_(self.ema_decay).add_(
                    cluster_size, alpha=1 - self.ema_decay)

                # 计算每个码字的向量之和
                dw = one_hot.t() @ z_flat  # [num_emb, D]
                # EMA 更新 ema_w
                self.ema_w.data.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

                # Laplace 平滑
                n = self.ema_cluster_size.sum()
                smoothed_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                self.codebook.weight.data.copy_(self.ema_w / smoothed_size.unsqueeze(1))

                # 更新使用计数
                self.usage_count.data += cluster_size.long()

                # Dead code revival: 重置长期未使用的码字
                if self.training and (self.usage_count.max() > self.dead_code_threshold):
                    dead_mask = self.usage_count < 1
                    n_dead = dead_mask.sum().item()
                    if n_dead > 0:
                        # 随机采样当前 batch 的数据点替换 dead code
                        n_replace = min(n_dead, z_flat.shape[0])
                        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:n_replace]
                        rand_indices = torch.randperm(z_flat.shape[0], device=z_flat.device)[:n_replace]
                        self.codebook.weight.data[dead_indices] = z_flat[rand_indices].detach()
                        self.ema_w.data[dead_indices] = z_flat[rand_indices].detach()
                        self.usage_count.data[dead_indices] = 1

        return quantized_st, indices_2d, loss


class VQVAE(nn.Module):
    """完整 VQ-VAE 模型：Encoder + Quantizer + Decoder。

    Args:
        downsample: 下采样倍数（4, 8, 16）。
            - 4: 32x32 → 8x8=64 tokens（信息充分，重建质量好）
            - 8: 32x32 → 4x4=16 tokens（平衡，默认）
            - 16: 32x32 → 2x2=4 tokens（信息不足，仅用于大图像）
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        num_embeddings: int = 8192,
        commitment_cost: float = 0.25,
        downsample: int = 8,
    ):
        super().__init__()
        self.downsample_factor = downsample
        self.encoder = VQVAEEncoder(in_channels, hidden_dim, latent_dim, downsample=downsample)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = VQVAEDecoder(in_channels, hidden_dim, latent_dim, downsample=downsample)

    def forward(self, x: torch.Tensor):
        """[B, 3, H, W] → 重建图像 + indices + loss"""
        z = self.encoder(x)
        quantized, indices, vq_loss = self.quantizer(z)
        recon = self.decoder(quantized)
        return recon, indices, vq_loss

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] → [B, H', W'] codebook 索引"""
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """[B, H', W'] codebook 索引 → [B, 3, H, W] 重建图像"""
        quantized = self.quantizer.codebook(indices)  # [B, H', W', D]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # [B, D, H', W']
        return self.decoder(quantized)


class VQVAEImageCodec:
    """图像编解码器 — 封装 VQ-VAE，满足 TokenizerHub 接口契约。

    与 TokenizerHub 的多模态接口对齐：
        codec.encode(image: torch.Tensor) -> list[int]   # codebook 索引序列
        codec.decode(ids: list[int]) -> torch.Tensor      # 重建图像
        codec.vocab_size() -> int                          # 8192
        codec.eos_token_id() -> int                        # -1

    Usage:
        codec = VQVAEImageCodec()
        hub.register_modality("image", codec)
        ids = hub.encode(image_tensor, domain="general", modality="image")
        recon = hub.decode(ids, domain="general", modality="image")
    """

    def __init__(
        self,
        model: Optional[VQVAE] = None,
        image_size: int = 224,
        device: Optional[torch.device] = None,
        downsample: int = 8,
    ):
        """Args:
            model: 预训练 VQ-VAE 模型（None 时按 downsample 创建未训练实例）
            image_size: 输入图像尺寸（默认 224×224）
            device: torch 设备
            downsample: 下采样倍数（4/8/16），仅在 model=None 时生效
        """
        self.model = model or VQVAE(downsample=downsample)
        self.model.eval()
        self.image_size = image_size
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        # 记录下采样倍数，用于 decode 时还原空间尺寸
        self.downsample_factor = getattr(self.model, "downsample_factor", downsample)

        # 从 config.py 读取 codebook size（与 tokenizer_contract.json 对齐）
        self._codebook_size = MULTIMODAL_TOKENS["image_codebook_size"]  # 8192

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """预处理：resize + 归一化到 [0, 1]。"""
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [3, H, W] → [1, 3, H, W]
        # Resize 到 image_size × image_size
        if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear")
        # 归一化到 [0, 1]
        if image.dtype != torch.float32:
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        return image.to(self.device)

    def encode(self, image: torch.Tensor) -> List[int]:
        """图像 → codebook 索引序列。

        Args:
            image: [3, H, W] 或 [B, 3, H, W]，值范围 [0, 1] 或 [0, 255]

        Returns:
            codebook 索引列表（0 ~ 8191）
        """
        with torch.no_grad():
            x = self._preprocess(image)
            indices = self.model.encode_to_indices(x)  # [B, H', W']
            # 展平为 1D 列表（batch=0）
            return indices[0].flatten().tolist()

    def decode(self, ids: List[int]) -> torch.Tensor:
        """codebook 索引序列 → 重建图像。

        Args:
            ids: codebook 索引列表（0 ~ 8191）

        Returns:
            重建图像 [3, H, W]（值范围 [0, 1]）
        """
        with torch.no_grad():
            # 计算空间尺寸（正方形）
            n = len(ids)
            h = w = int(n ** 0.5)
            if h * w != n:
                # 非正方形，用最长边
                h = int(n ** 0.5) + 1
                w = (n + h - 1) // h
            indices = torch.tensor(ids, dtype=torch.long, device=self.device).view(1, h, w)
            recon = self.model.decode_from_indices(indices)  # [1, 3, H, W]
            return recon[0].clamp(0, 1)

    def vocab_size(self) -> int:
        """返回 codebook 大小。"""
        return self._codebook_size

    def eos_token_id(self) -> int:
        """图像无 EOS 概念，返回 -1。"""
        return -1
