"""
Seed - 主入口
项目许可：Apache License 2.0，详见仓库根目录 LICENSE

统一入口点：
  python main.py                    # 启动 API 服务（模型+前端）
  python main.py --no-ui            # 仅加载模型（命令行模式）
  python main.py --train            # 训练模式
"""
import argparse
import os
import sys

# Windows 终端 UTF-8 支持
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 确保项目根目录在 Python 路径中
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from neuroplex.core.config import TrainingConfig, get_config, save_config
import uvicorn


def main():
    """主入口"""
    # 第一层：主入口参数
    parser = argparse.ArgumentParser(description="Seed")
    parser.add_argument("--model_name", type=str, default=None,
                        help="模型名称或路径")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="缓存目录")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="检查点路径")
    parser.add_argument("--no-ui", action="store_true",
                        help="仅加载模型，不启动 UI")
    parser.add_argument("--train", action="store_true",
                        help="启动训练模式")
    
    # 解析主入口参数并移除已解析的，避免传递给 get_config
    args, remaining = parser.parse_known_args()

    # 使用剩余参数（如果有）或 None 来获取训练配置
    if remaining:
        config = get_config(args=remaining)
    else:
        config = get_config(args=[])

    if args.model_name:
        config.model_name = args.model_name
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    if args.checkpoint:
        config.resume_from_checkpoint = args.checkpoint

    print("🧠 Seed")
    print(f"   模型: {config.model_name}")
    print(f"   设备: {config.resolve_device()}")

    if args.no_ui:
        print("ℹ️ no-ui 模式：仅 Cortex 加载")
        from neuroplex.loader import load_cortex
        cortex, tokenizer = load_cortex(
            neurons_dir=config.model_name or "data/neurons",
            device=config.resolve_device(),
        )
        print(f"✅ Cortex 已加载: {len(cortex.neurons)} 个神经元")
        return

    if args.train:
        print("🚀 训练模式...")
        print("   Cortex 神经元架构训练方式：")
        print("   1. 对话神经元 SFT: scripts/training/finetune_neuron_dialogue.py")
        print("   2. 协作层微调: scripts/training/finetune_cross_spec.py / finetune_side_channels.py")
        print("   3. 跨域协作联合训练（含 hub）: scripts/training/train_cross_domain_collab.py")
        print("   4. hub 神经元从零训练: scripts/training/train_hub_neuron.py")
        print("   5. 运行时学习: feed_engine + sleep_engine（在线闭环）")
        print()
        print("   查看脚本帮助：")
        print("   python scripts/training/finetune_neuron_dialogue.py --help")
        return

    print("🚀 正在通过命令行启动后台 API 服务...")
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
