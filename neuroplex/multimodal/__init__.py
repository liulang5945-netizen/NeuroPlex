"""多模态模块 — 图像/音频/视频编解码器。

P8: 实现三个具体模态编码器，让 register_modality 有真实实现。

接口契约（与 TokenizerHub 对齐）：
    codec.encode(raw_input) -> list[int]      # 离散化为 token id 序列
    codec.decode(ids: list[int]) -> raw_output
    codec.vocab_size() -> int
    codec.eos_token_id() -> int  (可选，无则返回 -1)

模态清单：
    image: VQ-VAE (codebook=8192, 下采样 16x)
    audio: EnCodec (codebook=4096, 下采样 128x)
    video: 3D CNN VQ-VAE (复用 image codebook=8192, 空间 16x + 时间 4x)
"""
from .vqvae import VQVAEImageCodec, VQVAE
from .encodec import EnCodecAudioCodec, EnCodec
from .video import VideoCodec, VideoVQVAE
from .io import save_image, save_audio, save_video

__all__ = [
    "VQVAEImageCodec", "VQVAE",
    "EnCodecAudioCodec", "EnCodec",
    "VideoCodec", "VideoVQVAE",
    "save_image", "save_audio", "save_video",
]
