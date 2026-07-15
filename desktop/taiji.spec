# -*- mode: python ; coding: utf-8 -*-
"""
态极桌面客户端 PyInstaller 配置

使用方式：
    pyinstaller desktop/taiji.spec
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(SPECPATH)

block_cipher = None

a = Analysis(
    [str(ROOT_DIR / 'desktop' / 'main.py')],
    pathex=[str(ROOT_DIR)],
    binaries=[],
    datas=[
        (str(ROOT_DIR / 'frontend' / 'dist'), 'frontend/dist'),
        (str(ROOT_DIR / 'taiji_data' / 'final'), 'taiji_data/final'),
        (str(ROOT_DIR / 'app_settings.json'), '.'),
        (str(ROOT_DIR / 'version.json'), '.'),
        (str(ROOT_DIR / 'icon.ico'), '.'),
    ],
    hiddenimports=[
        'taiji', 'taiji.core', 'taiji.core.api', 'taiji.core.inference',
        'taiji.core.app_state', 'taiji.body', 'taiji.body.core',
        'taiji.brain', 'taiji.brain.cortex',
        'taiji.life', 'taiji.life.life_scheduler',
        'taiji.life.feed_engine', 'taiji.life.sleep_engine',
        'taiji.life.play_engine', 'taiji.life.evolution_engine',
        'taiji.life.explore_engine', 'taiji.life.science_engine',
        'taiji.agent', 'taiji.agent.working_memory',
        'taiji.agent.context_manager', 'taiji.agent.semantic_memory',
        'taiji.agent_ext', 'taiji.agent_ext.react_engine',
        'taiji.agent_ext.tool_registry', 'taiji.agent_ext.agent',
        'taiji.tools', 'taiji.tools.web', 'taiji.tools.rag',
        'taiji.tools.desktop', 'taiji.tools.searxng', 'taiji.tools.browser',
        'taiji.safety', 'taiji.safety.safety',
        'taiji.infra', 'taiji.infra.events',
        'api', 'api.app', 'api.routes_chat', 'api.routes_taiji',
        'api.routes_life', 'api.routes_agent', 'api.routes_models',
        'api.routes_training', 'api.routes_settings', 'api.routes_rag',
        'api.chat_strategies', 'api.models',
        'uvicorn', 'uvicorn.logging', 'uvicorn.loops',
        'uvicorn.loops.auto', 'uvicorn.protocols',
        'uvicorn.protocols.http', 'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan', 'uvicorn.lifespan.on',
        'fastapi', 'pydantic', 'pydantic.deprecated',
        'torch', 'transformers', 'sentence_transformers',
        'sentencepiece', 'numpy', 'jieba',
        'PyQt6', 'PyQt6.QtWidgets', 'PyQt6.QtCore', 'PyQt6.QtGui',
        'PyQt6.QtWebEngineWidgets', 'PyQt6.QtWebEngineCore',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'pandas', 'tensorboard', 'datasets',
        'langchain', 'langchain_core', 'langchain_community',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Taiji',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # 无控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT_DIR / 'icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Taiji',
)
