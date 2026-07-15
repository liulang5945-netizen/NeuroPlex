"""
态极桌面客户端打包脚本
========================

使用 PyInstaller 打包为独立可执行文件。

使用方式：
    python desktop/build.py

输出：
    dist/Taiji.exe (Windows)
    dist/Taiji (Linux/Mac)
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DIST_DIR = ROOT_DIR / "dist"
BUILD_DIR = ROOT_DIR / "build"


def build():
    """打包态极桌面客户端"""
    print("=" * 50)
    print("  态极桌面客户端打包")
    print("=" * 50)

    # 清理旧构建
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    # 前端构建
    print("\n[1/3] 构建前端...")
    frontend_dir = ROOT_DIR / "frontend"
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=str(frontend_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"前端构建失败: {result.stderr}")
        return False
    print("  前端构建完成")

    # PyInstaller 打包
    print("\n[2/3] PyInstaller 打包...")

    # 收集数据文件
    datas = [
        (str(ROOT_DIR / "frontend" / "dist"), "frontend/dist"),
        (str(ROOT_DIR / "taiji_data" / "final"), "taiji_data/final"),
        (str(ROOT_DIR / "app_settings.json"), "."),
        (str(ROOT_DIR / "version.json"), "."),
        (str(ROOT_DIR / "icon.ico"), "."),
    ]

    # 构建 PyInstaller 命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=Taiji",
        "--windowed",  # 无控制台窗口
        "--onedir",    # 单目录打包
        f"--icon={ROOT_DIR / 'icon.ico'}",
        "--noconfirm",
    ]

    # 添加数据文件
    for src, dst in datas:
        if Path(src).exists():
            cmd.append(f"--add-data={src};{dst}")

    # 添加隐式导入
    hidden_imports = [
        "taiji", "api", "uvicorn", "fastapi", "pydantic",
        "torch", "transformers", "sentence_transformers",
        "PyQt6", "PyQt6.QtWebEngineWidgets",
    ]
    for imp in hidden_imports:
        cmd.append(f"--hidden-import={imp}")

    # 入口点
    cmd.append(str(ROOT_DIR / "desktop" / "main.py"))

    print(f"  命令: {' '.join(cmd[:5])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"打包失败:\n{result.stderr[-500:]}")
        return False

    print("  PyInstaller 打包完成")

    # 后处理
    print("\n[3/3] 后处理...")

    # 复制额外文件到 dist
    dist_taiji = DIST_DIR / "Taiji"
    if dist_taiji.exists():
        # 复制知识库目录
        for extra_dir in ["knowledge_store", "user_data", "security"]:
            src = ROOT_DIR / extra_dir
            if src.exists():
                shutil.copytree(src, dist_taiji / extra_dir, dirs_exist_ok=True)

        # 创建空目录
        for empty_dir in ["agent_workspace", "taiji_data/feed_data", "taiji_data/sleep_data",
                          "taiji_data/life_data", "taiji_data/evolution_data"]:
            (dist_taiji / empty_dir).mkdir(parents=True, exist_ok=True)

    print("  后处理完成")

    # 统计
    print("\n" + "=" * 50)
    total_size = sum(f.stat().st_size for f in DIST_DIR.rglob("*") if f.is_file())
    print(f"  输出目录: {DIST_DIR}")
    print(f"  总大小: {total_size / 1024 / 1024:.1f} MB")
    print("=" * 50)

    return True


if __name__ == "__main__":
    success = build()
    sys.exit(0 if success else 1)
