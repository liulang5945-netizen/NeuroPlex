"""P8: 训练 VQ-VAE 图像编解码器。

让 VQVAEImageCodec 具备实际重建能力，为后续图像生成 neuron 的 sleep 训练提供预训练 codec。

用法:
    # 合成数据快速验证（无外部依赖）
    python scripts/training/train_vqvae.py --steps 500 --image-size 64

    # CIFAR-10 真实图像训练（自动下载）
    python scripts/training/train_vqvae.py --data-source cifar10 --steps 3000 --image-size 32

    # 真实图像目录训练
    python scripts/training/train_vqvae.py --data-dir data/images --steps 5000 --image-size 128

    # 加载已有 checkpoint 继续训练
    python scripts/training/train_vqvae.py --resume data/vqvae/vqvae_latest.pt --steps 2000

    # 评估 PSNR（不训练）
    python scripts/training/train_vqvae.py --eval-only --resume data/vqvae/vqvae_latest.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import math
import pickle
import tarfile
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.multimodal.vqvae import VQVAE

# ── 默认超参数 ──
BATCH_SIZE = 16
IMAGE_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
DEFAULT_STEPS = 2000
LOG_INTERVAL = 50
SAVE_INTERVAL = 500
NUM_EMBEDDINGS = 8192          # 与 tokenizer_contract.json 对齐
LATENT_DIM = 256
HIDDEN_DIM = 128
COMMITMENT_COST = 0.25

OUTPUT_DIR = "data/vqvae"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_DIR = "data/cifar-10-batches-py"


def synthesize_batch(batch_size: int, image_size: int, device: torch.device) -> torch.Tensor:
    """合成图像 batch（无外部数据依赖，用于验证训练路径）。

    生成带结构和纹理的多样化图像：
    - 渐变背景（线性/径向/对角）
    - 多种几何形状（圆、方、三角、线条）
    - 多种纹理（棋盘、条纹、噪声）
    - 随机颜色组合

    比 CIFAR-10 简单但足够多样，能让 VQ-VAE 学到有用 codebook。
    """
    yy, xx = torch.meshgrid(
        torch.arange(image_size, device=device, dtype=torch.float32),
        torch.arange(image_size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xx_n = xx / image_size  # 归一化到 [0, 1]
    yy_n = yy / image_size

    batch = []
    for _ in range(batch_size):
        # 随机选择背景类型
        bg_type = torch.randint(0, 4, (1,)).item()

        # 随机基色
        base_color = torch.rand(3, 1, 1, device=device) * 0.6 + 0.2  # [0.2, 0.8]

        if bg_type == 0:
            # 线性渐变
            direction = torch.rand(1).item() * 3.14159
            grad = (xx_n * torch.cos(torch.tensor(direction)) +
                    yy_n * torch.sin(torch.tensor(direction)))
            grad = grad / grad.max().clamp(min=0.01)
            img = base_color * grad
        elif bg_type == 1:
            # 径向渐变
            cx, cy = torch.rand(2, device=device) * image_size
            dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            dist = 1.0 - (dist / dist.max().clamp(min=1)).clamp(0, 1)
            img = base_color * dist
        elif bg_type == 2:
            # 棋盘格
            cell_size = torch.randint(4, 16, (1,)).item()
            checker = ((xx // cell_size + yy // cell_size) % 2).float()
            img = base_color * (0.3 + 0.7 * checker)
        else:
            # 条纹
            stripe_width = torch.randint(2, 10, (1,)).item()
            stripe = (xx // stripe_width) % 2
            img = base_color * (0.4 + 0.6 * stripe.float())

        img = img.clamp(0, 1)

        # 随机叠加 1-3 个几何形状
        n_shapes = torch.randint(1, 4, (1,)).item()
        for _ in range(n_shapes):
            shape_type = torch.randint(0, 3, (1,)).item()
            cx = torch.rand(1, device=device) * image_size
            cy = torch.rand(1, device=device) * image_size
            size = torch.rand(1, device=device) * (image_size / 3) + 5
            color = torch.rand(3, 1, 1, device=device)

            if shape_type == 0:
                # 圆形
                mask = ((xx - cx) ** 2 + (yy - cy) ** 2 < size ** 2).float()
            elif shape_type == 1:
                # 矩形
                mask = ((xx > cx - size) & (xx < cx + size) &
                        (yy > cy - size) & (yy < cy + size)).float()
            else:
                # 线条
                angle = torch.rand(1).item() * 3.14159
                dx = xx - cx
                dy = yy - cy
                proj = dx * torch.cos(torch.tensor(angle)) + dy * torch.sin(torch.tensor(angle))
                perp = -dx * torch.sin(torch.tensor(angle)) + dy * torch.cos(torch.tensor(angle))
                mask = (proj.abs() < size) & (perp.abs() < 2)
                mask = mask.float()

            img = img * (1 - mask) + color * mask

        # 随机加少量噪声（低 std 避免 PSNR 上限受限）
        if torch.rand(1).item() > 0.7:
            noise = torch.randn(3, image_size, image_size, device=device) * 0.02
            img = (img + noise).clamp(0, 1)

        batch.append(img)

    return torch.stack(batch)  # [B, 3, H, W]


def load_real_images(
    data_dir: str,
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    """从目录加载真实图像（简单实现，无 torchvision 依赖）。

    支持 .png/.jpg/.jpeg，用 PIL 加载并 resize。
    """
    try:
        from PIL import Image
        import random
    except ImportError:
        raise ImportError("加载真实图像需要 Pillow: pip install Pillow")

    files = []
    for root, _, fnames in os.walk(data_dir):
        for fname in fnames:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                files.append(os.path.join(root, fname))

    if not files:
        raise FileNotFoundError(f"目录 {data_dir} 中无图像文件")

    batch = []
    for _ in range(batch_size):
        path = random.choice(files)
        img = Image.open(path).convert("RGB").resize((image_size, image_size))
        arr = torch.tensor(list(img.getdata()), dtype=torch.float32, device=device)
        arr = arr.view(image_size, image_size, 3).permute(2, 0, 1) / 255.0
        batch.append(arr)

    return torch.stack(batch)


def download_cifar10():
    """下载并解压 CIFAR-10（如果尚未存在）。"""
    dest = "data/cifar-10-python.tar.gz"
    if not os.path.exists(CIFAR10_DIR):
        if not os.path.exists(dest):
            print(f"Downloading CIFAR-10 from {CIFAR10_URL}...")
            os.makedirs("data", exist_ok=True)
            urllib.request.urlretrieve(CIFAR10_URL, dest)
            print(f"Downloaded: {os.path.getsize(dest)/1e6:.1f}MB")
        print("Extracting CIFAR-10...")
        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall("data/")
        print(f"CIFAR-10 extracted to {CIFAR10_DIR}")


def load_cifar10_batch(batch_size: int, device: torch.device) -> torch.Tensor:
    """从 CIFAR-10 随机加载一个 batch（32x32 真实图像）。

    Returns:
        [B, 3, 32, 32] tensor in [0, 1]
    """
    if not os.path.exists(CIFAR10_DIR):
        download_cifar10()

    # 随机选一个 batch 文件
    import random
    batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    batch_file = random.choice(batch_files)
    with open(os.path.join(CIFAR10_DIR, batch_file), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    imgs = batch[b"data"]  # [N, 3072] uint8

    # 随机采样 batch_size 张
    indices = random.sample(range(len(imgs)), batch_size)
    selected = imgs[indices]  # [B, 3072]
    # 重塑为 [B, 3, 32, 32]
    batch_tensor = torch.tensor(selected, dtype=torch.float32, device=device)
    batch_tensor = batch_tensor.view(batch_size, 3, 32, 32) / 255.0
    return batch_tensor


def compute_psnr(model: VQVAE, device: torch.device, n_samples: int = 100,
                 data_source: str = "cifar10", image_size: int = 32) -> float:
    """评估模型重建质量（PSNR dB）。

    PSNR > 25dB = 可用重建
    PSNR > 30dB = 良好重建
    PSNR < 20dB = 噪声级别
    """
    model.eval()
    total_mse = 0.0
    count = 0
    batch_size = 10

    with torch.no_grad():
        for _ in range(n_samples // batch_size):
            if data_source == "cifar10":
                batch = load_cifar10_batch(batch_size, device)
            else:
                batch = synthesize_batch(batch_size, image_size, device)

            recon, _, _ = model(batch)
            mse = F.mse_loss(recon, batch).item()
            total_mse += mse * batch_size
            count += batch_size

    avg_mse = total_mse / count
    if avg_mse < 1e-10:
        return 100.0
    psnr = 10 * math.log10(1.0 / avg_mse)
    model.train()
    return psnr


def train(
    steps: int,
    batch_size: int,
    image_size: int,
    lr: float,
    data_dir: str | None,
    output_dir: str,
    resume: str | None,
    device: torch.device,
    log_interval: int = LOG_INTERVAL,
    save_interval: int = SAVE_INTERVAL,
    data_source: str = "synthetic",  # synthetic / cifar10 / dir
    eval_psnr: bool = True,
    downsample: int = 4,
):
    """训练 VQ-VAE。"""
    print(f"VQ-VAE 训练配置:")
    print(f"  steps={steps}, batch_size={batch_size}, image_size={image_size}")
    print(f"  lr={lr}, device={device}, downsample={downsample}")
    print(f"  data_source={data_source}" + (f", dir={data_dir}" if data_dir else ""))
    print(f"  output={output_dir}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建模型
    model = VQVAE(
        in_channels=3,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_embeddings=NUM_EMBEDDINGS,
        commitment_cost=COMMITMENT_COST,
        downsample=downsample,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.1f}M")

    # 2. 加载 checkpoint（若指定）
    start_step = 0
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"从 checkpoint 恢复: {resume} (step={start_step})")

    model.train()

    # 3. 优化器（排除 codebook 参数，codebook 由 EMA 更新）
    codebook_params = set()
    for mod in model.modules():
        if hasattr(mod, 'codebook') and isinstance(mod.codebook, nn.Embedding):
            codebook_params.add(id(mod.codebook.weight))
    trainable = [p for p in model.parameters() if id(p) not in codebook_params]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)

    # 4. 训练循环
    print(f"\n开始训练（step {start_step} → {steps}）...")
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    log_count = 0
    start_time = time.time()

    for step in range(start_step, steps):
        # 4.1 获取 batch
        if data_source == "cifar10":
            try:
                batch = load_cifar10_batch(batch_size, device)
                # CIFAR-10 是 32x32，如果 image_size 不同则 resize
                if image_size != 32:
                    batch = F.interpolate(batch, size=(image_size, image_size), mode="bilinear")
            except Exception as e:
                print(f"  Warning: CIFAR-10 加载失败 ({e})，回退到合成数据")
                batch = synthesize_batch(batch_size, image_size, device)
        elif data_source == "dir" and data_dir is not None:
            try:
                batch = load_real_images(data_dir, batch_size, image_size, device)
            except Exception as e:
                print(f"  Warning: 加载真实图像失败 ({e})，回退到合成数据")
                batch = synthesize_batch(batch_size, image_size, device)
        else:
            batch = synthesize_batch(batch_size, image_size, device)

        # 4.2 Forward
        optimizer.zero_grad()
        recon, indices, vq_loss = model(batch)

        # 4.3 Loss: reconstruction (MSE) + VQ (codebook + commitment)
        recon_loss = F.mse_loss(recon, batch)
        loss = recon_loss + vq_loss

        # 4.4 Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        log_count += 1

        # 4.5 日志
        if (step + 1) % log_interval == 0:
            avg_recon = total_recon_loss / log_count
            avg_vq = total_vq_loss / log_count
            elapsed = time.time() - start_time
            steps_per_sec = log_count / elapsed
            # codebook 使用率（避免 dead code）
            unique_codes = len(torch.unique(indices))
            utilization = unique_codes / NUM_EMBEDDINGS * 100

            psnr_str = ""
            if eval_psnr and (step + 1) % (log_interval * 4) == 0:
                psnr = compute_psnr(model, device, n_samples=50,
                                    data_source=data_source, image_size=image_size)
                psnr_str = f" | PSNR={psnr:.1f}dB"
                model.train()

            print(f"  step {step + 1}/{steps} | "
                  f"recon={avg_recon:.4f} vq={avg_vq:.4f} | "
                  f"codebook: {unique_codes}/{NUM_EMBEDDINGS} ({utilization:.1f}%){psnr_str} | "
                  f"{steps_per_sec:.1f} steps/s")

            total_recon_loss = 0.0
            total_vq_loss = 0.0
            log_count = 0
            start_time = time.time()

        # 4.6 保存 checkpoint
        if (step + 1) % save_interval == 0 or step + 1 == steps:
            ckpt_path = os.path.join(output_dir, "vqvae_latest.pt")
            # 保存前评估 PSNR
            final_psnr = 0.0
            if eval_psnr:
                final_psnr = compute_psnr(model, device, n_samples=100,
                                          data_source=data_source, image_size=image_size)
                model.train()
            torch.save({
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "config": {
                    "image_size": image_size,
                    "num_embeddings": NUM_EMBEDDINGS,
                    "latent_dim": LATENT_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "commitment_cost": COMMITMENT_COST,
                    "downsample": downsample,
                },
                "psnr": final_psnr,
            }, ckpt_path)
            print(f"  💾 checkpoint saved: {ckpt_path} (PSNR={final_psnr:.1f}dB)")

    print(f"\n训练完成！checkpoint: {os.path.join(output_dir, 'vqvae_latest.pt')}")


def main():
    parser = argparse.ArgumentParser(description="训练 VQ-VAE 图像编解码器")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"训练步数（默认 {DEFAULT_STEPS}）")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"batch size（默认 {BATCH_SIZE}）")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                        help=f"图像尺寸（默认 {IMAGE_SIZE}）")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"学习率（默认 {LR}）")
    parser.add_argument("--data-source", type=str, default="synthetic",
                        choices=["synthetic", "cifar10", "dir"],
                        help="数据源：synthetic(合成) / cifar10(自动下载) / dir(目录)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="真实图像目录（仅 data-source=dir 时生效）")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"输出目录（默认 {OUTPUT_DIR}）")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备（auto/cpu/cuda）")
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL,
                        help=f"日志间隔（默认 {LOG_INTERVAL}）")
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL,
                        help=f"保存间隔（默认 {SAVE_INTERVAL}）")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估 PSNR（不训练）")
    parser.add_argument("--no-psnr", action="store_true",
                        help="禁用训练中 PSNR 评估（加速）")
    parser.add_argument("--downsample", type=int, default=4, choices=[4, 8, 16],
                        help="下采样倍数（4=高重建质量/多 token，8=平衡，16=少 token）默认 4")

    args = parser.parse_args()

    # 设备选择
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 仅评估模式
    if args.eval_only:
        if not args.resume or not os.path.exists(args.resume):
            print("Error: --eval-only 需要 --resume 指定 checkpoint")
            return
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        cfg_dict = ckpt.get("config", {})
        model = VQVAE(
            hidden_dim=cfg_dict.get("hidden_dim", HIDDEN_DIM),
            latent_dim=cfg_dict.get("latent_dim", LATENT_DIM),
            num_embeddings=cfg_dict.get("num_embeddings", NUM_EMBEDDINGS),
            commitment_cost=cfg_dict.get("commitment_cost", COMMITMENT_COST),
            downsample=cfg_dict.get("downsample", 16),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        ds = "cifar10" if args.data_source == "cifar10" else "synthetic"
        psnr = compute_psnr(model, device, n_samples=200,
                            data_source=ds, image_size=args.image_size)
        print(f"PSNR: {psnr:.2f} dB (checkpoint: {args.resume}, step={ckpt.get('step', '?')})")
        print(f"  {'✅ 可用重建 (>25dB)' if psnr > 25 else '⚠ 噪声级别 (<20dB)' if psnr < 20 else '🔧 待提升 (20-25dB)'}")
        return

    train(
        steps=args.steps,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resume=args.resume,
        device=device,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        data_source=args.data_source,
        eval_psnr=not args.no_psnr,
        downsample=args.downsample,
    )


if __name__ == "__main__":
    main()
