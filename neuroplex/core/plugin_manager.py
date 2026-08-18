"""
插件管理器 (Plugin Manager)
============================
支持插件发现、加载、卸载、工具注册、路由注册。
"""
import importlib
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("PluginManager")


@dataclass
class PluginManifest:
    """插件清单"""
    id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    enabled: bool = True
    tools: List[dict] = field(default_factory=list)
    entry_point: str = "__init__.py"
    dependencies: List[str] = field(default_factory=list)


class PluginManager:
    """插件管理器"""

    def __init__(self, plugins_dir: str = ""):
        if not plugins_dir:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            plugins_dir = os.path.join(base_dir, "plugins")
        self._dir = plugins_dir
        os.makedirs(self._dir, exist_ok=True)
        self._plugins: Dict[str, PluginManifest] = {}
        self._loaded_modules: Dict[str, object] = {}
        self._discover()

    def _discover(self):
        """扫描插件目录发现插件，并自动加载 enabled 的插件"""
        for item in os.listdir(self._dir):
            manifest_path = os.path.join(self._dir, item, "manifest.json")
            if os.path.isfile(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    manifest = PluginManifest(**{k: v for k, v in data.items() if k in PluginManifest.__dataclass_fields__})
                    manifest.id = manifest.id or item
                    self._plugins[manifest.id] = manifest
                    logger.info(f"发现插件: {manifest.name} v{manifest.version}")
                except Exception as e:
                    logger.warning(f"插件 {item} 清单解析失败: {e}")

        # 自动加载所有 enabled 的插件
        for plugin_id, manifest in list(self._plugins.items()):
            if manifest.enabled:
                try:
                    self.load_plugin(plugin_id)
                except Exception as e:
                    logger.warning(f"自动加载插件 {plugin_id} 失败: {e}")

    def list_plugins(self) -> List[dict]:
        """列出所有插件"""
        return [
            {
                "id": p.id, "name": p.name, "version": p.version,
                "description": p.description, "author": p.author,
                "enabled": p.enabled, "tools_count": len(p.tools),
            }
            for p in self._plugins.values()
        ]

    def get_plugin(self, plugin_id: str) -> Optional[PluginManifest]:
        return self._plugins.get(plugin_id)

    def load_plugin(self, plugin_id: str) -> bool:
        """加载插件"""
        manifest = self._plugins.get(plugin_id)
        if not manifest:
            raise ValueError(f"插件 {plugin_id} 不存在")

        plugin_dir = os.path.join(self._dir, plugin_id)
        entry_path = os.path.join(plugin_dir, manifest.entry_point)

        if not os.path.exists(entry_path):
            raise FileNotFoundError(f"插件入口文件不存在: {entry_path}")

        # 动态导入
        spec = importlib.util.spec_from_file_location(f"plugin_{plugin_id}", entry_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._loaded_modules[plugin_id] = module

        # 注册工具
        if hasattr(module, "register_tools"):
            try:
                module.register_tools()
                logger.info(f"插件 {plugin_id} 工具已注册")
            except Exception as e:
                logger.warning(f"插件 {plugin_id} 工具注册失败: {e}")

        # 注册路由
        if hasattr(module, "get_router"):
            try:
                router = module.get_router()
                try:
                    from api.app import app
                    app.include_router(router, prefix=f"/api/plugins/{plugin_id}")
                except ImportError:
                    logger.debug("FastAPI app 不可用，跳过插件 %s 路由注册", plugin_id)
                logger.info(f"插件 {plugin_id} 路由已注册")
            except Exception as e:
                logger.warning(f"插件 {plugin_id} 路由注册失败: {e}")

        manifest.enabled = True
        self._save_manifest(manifest)
        logger.info(f"插件 {plugin_id} 已加载")

    def unload_plugin(self, plugin_id: str):
        """卸载插件"""
        if plugin_id in self._loaded_modules:
            module = self._loaded_modules[plugin_id]
            if hasattr(module, "unregister_tools"):
                try:
                    module.unregister_tools()
                except Exception:
                    pass
            del self._loaded_modules[plugin_id]

        manifest = self._plugins.get(plugin_id)
        if manifest:
            manifest.enabled = False
            self._save_manifest(manifest)
        logger.info(f"插件 {plugin_id} 已卸载")

    def install_plugin(self, source_path: str) -> str:
        """安装插件（从本地路径复制到插件目录）"""
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"插件源路径不存在: {source_path}")

        manifest_path = os.path.join(source_path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise ValueError("插件目录中缺少 manifest.json")

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        plugin_id = data.get("id", os.path.basename(source_path))
        dest = os.path.join(self._dir, plugin_id)

        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(source_path, dest)

        manifest = PluginManifest(**{k: v for k, v in data.items() if k in PluginManifest.__dataclass_fields__})
        manifest.id = plugin_id
        self._plugins[plugin_id] = manifest
        logger.info(f"插件 {plugin_id} 已安装")
        return plugin_id

    def uninstall_plugin(self, plugin_id: str):
        """卸载并删除插件"""
        self.unload_plugin(plugin_id)
        plugin_dir = os.path.join(self._dir, plugin_id)
        if os.path.exists(plugin_dir):
            shutil.rmtree(plugin_dir)
        self._plugins.pop(plugin_id, None)
        logger.info(f"插件 {plugin_id} 已删除")

    def _save_manifest(self, manifest: PluginManifest):
        from dataclasses import asdict
        path = os.path.join(self._dir, manifest.id, "manifest.json")
        if os.path.exists(os.path.dirname(path)):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(manifest), f, indent=2, ensure_ascii=False)