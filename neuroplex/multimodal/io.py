"""多模态文件 I/O — tensor ↔ 文件转换。

把 codec 输出的 raw tensor 保存为实际媒体文件，让 VQ-VAE/EnCodec/VideoVQVAE
的重建结果可被外部查看。

支持：
    image: [3,H,W] tensor → PNG（PIL）
    audio: [samples] tensor → WAV（wave 标准库）

视频（MP4）需要 imageio-ffmpeg，未安装时降级为逐帧 PNG 序列。
"""
from __future__ import annotations

import os
from typing import Optional

import torch

# PIL 在 train_vqvae.py 中已是依赖（load_real_images 用到）


def save_image(tensor: torch.Tensor, path: str) -> str:
    """[3,H,W] 或 [H,W,3] tensor → PNG 文件。

    Args:
        tensor: 图像 tensor，值范围 [0,1] 或 [0,255]
        path: 输出文件路径（.png）

    Returns:
        保存的文件路径
    """
    from PIL import Image

    if tensor.dim() == 2:
        # 灰度 [H,W] → [1,H,W]
        tensor = tensor.unsqueeze(0).expand(3, -1, -1)

    if tensor.dim() == 3 and tensor.shape[-1] in (1, 3):
        # [H,W,3] → [3,H,W]
        tensor = tensor.permute(2, 0, 1)

    # [3,H,W] → [H,W,3]
    img = tensor.detach().cpu()
    if img.dtype != torch.float32:
        img = img.float()
    if img.max() > 1.0:
        img = img / 255.0
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0) * 255).byte().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    Image.fromarray(img).save(path)
    return path


def save_audio(tensor: torch.Tensor, path: str, sample_rate: int = 16000) -> str:
    """[samples] 或 [samples, channels] tensor → WAV 文件。

    Args:
        tensor: 音频 tensor，值范围 [-1,1]（float）或 [-32768,32767]（int）
        path: 输出文件路径（.wav）
        sample_rate: 采样率（默认 16000）

    Returns:
        保存的文件路径
    """
    import wave
    import struct

    audio = tensor.detach().cpu().float()
    if audio.dim() == 2:
        # [samples, channels] → mono by averaging
        audio = audio.mean(dim=1)

    # 归一化到 [-1, 1]
    max_val = audio.abs().max()
    if max_val > 0:
        audio = audio / max_val

    # 转 16-bit PCM
    pcm = (audio * 32767).clamp(-32768, 32767).short().tolist()

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))
    return path


def save_video(
    tensor: torch.Tensor,
    path: str,
    fps: int = 8,
    fallback_png: bool = True,
) -> str:
    """[T,C,H,W] tensor → MP4 文件（需 imageio-ffmpeg）。

    Args:
        tensor: 视频帧序列 [T,C,H,W]，值范围 [0,1] 或 [0,255]
        path: 输出文件路径（.mp4）
        fps: 帧率（默认 8）
        fallback_png: 若 imageio 不可用，降级为逐帧 PNG（返回目录路径）

    Returns:
        保存的文件/目录路径
    """
    frames = tensor.detach().cpu().float()
    if frames.dtype != torch.float32:
        frames = frames.float()
    if frames.max() > 1.0:
        frames = frames / 255.0
    frames = frames.clamp(0, 1)

    # [T,C,H,W] → [T,H,W,C]
    frames = frames.permute(0, 2, 3, 1)
    frames_np = (frames * 255).byte().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(path, fps=fps, codec="libx264")
        for frame in frames_np:
            writer.append_data(frame)
        writer.close()
        return path
    except (ImportError, RuntimeError):
        if not fallback_png:
            raise
        # 降级：逐帧 PNG
        base, _ = os.path.splitext(path)
        out_dir = base + "_frames"
        os.makedirs(out_dir, exist_ok=True)
        from PIL import Image
        for i, frame in enumerate(frames_np):
            Image.fromarray(frame).save(os.path.join(out_dir, f"frame_{i:04d}.png"))
        return out_dir


__all__ = ["save_image", "save_audio", "save_video", "load_image", "load_audio", "load_video"]


# ── Load 函数 ──


def load_image(path: str) -> torch.Tensor:
    """从文件加载图像 → [3,H,W] tensor (0~1)。"""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(_pil_to_np(img)).permute(2, 0, 1).float() / 255.0


def load_audio(path: str, sample_rate: int = 16000) -> torch.Tensor:
    """从文件加载音频 → [1, samples] tensor (-1~1)。

    优先用 soundfile，否则用 wave 标准库。
    """
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]  # 取左声道
        import librosa
        if sr != sample_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
        return torch.from_numpy(data).unsqueeze(0)
    except ImportError:
        import wave
        import numpy as np
        with wave.open(path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
        dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
        np_dtype = dtype_map.get(sampwidth, np.int16)
        audio = np.frombuffer(frames, dtype=np_dtype).astype(np.float32)
        if nchannels > 1:
            audio = audio[::nchannels]
        audio = audio / max(abs(audio).max(), 1.0)
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        return torch.from_numpy(audio).unsqueeze(0)


def load_video(path: str) -> torch.Tensor:
    """从文件加载视频 → [3,T,H,W] tensor (0~1)。

    用 imageio 逐帧读取，失败时尝试 torchvision。
    """
    import numpy as np
    try:
        import imageio.v2 as imageio
        reader = imageio.get_reader(path)
        frames = []
        for frame in reader:
            frames.append(frame)
        reader.close()
        # [T, H, W, 3] → [3, T, H, W]
        video = np.stack(frames).astype(np.float32) / 255.0
        return torch.from_numpy(video).permute(3, 0, 1, 2)
    except ImportError:
        raise ImportError("需要 imageio 来加载视频: pip install imageio")


def _pil_to_np(img):
    """PIL Image → numpy [H,W,3]。"""
    import numpy as np
    return np.array(img)
