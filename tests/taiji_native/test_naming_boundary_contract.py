"""命名与架构边界的可执行守护。

这些测试把 ARCHITECTURE_DIRECTION_2026_08.md §0 规范词表从"文档约定"变成
"机器强制约束"。文档只能建议，测试才能阻止回退。

守护的四条不可回退事实：
  1. Seed 是项目/模型，seed/ 只能通过公开 API 组合 taiji/
  2. taiji/ 是自足基底：不导入 seed、neuroplex 或 transformers（含子包）
  3. neuroplex/ 不反向依赖 seed/ 或 taiji/：冻结基线保持独立
  4. Transformer 底层的 live 消费点是封闭且已知的：新增消费点必须显式改这里
  5. taiji/ 内禁止出现"态极"：新基底不复用 Legacy NeuroPlex 的旧中文称呼
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Transformer 底层（neuroplex/layers.py::TransformerBlock）当前的 live 消费点。
# scripts/archive/ 不计入：那里是冻结的历史层，判定见 scripts/archive/README.md。
# 新增任何一条都必须先回答"为什么还要在被替代的底层上继续投入"。
LEGACY_TRANSFORMER_CONSUMERS = {
    "neuroplex/resonance/neuron.py",
    "scripts/training/train_tinystories.py",
    "scripts/training/train_tinystories_field.py",
}


def _iter_python_files(package: str) -> list[Path]:
    return sorted(
        path
        for path in (REPO / package).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _read(path: Path) -> str:
    """用 utf-8-sig：api/ 下有 3 个文件带 UTF-8 BOM（import 时无害，
    但按 utf-8 读会把 BOM 留在首字符，导致 ast.parse 失败）。"""
    return path.read_text(encoding="utf-8-sig")


def _parse(path: Path) -> ast.Module:
    return ast.parse(_read(path), filename=str(path))


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def _imports_symbol(path: Path, symbol: str) -> bool:
    """按 import 语句判定，而非文本子串。

    这样定义处（neuroplex/layers.py）和只在注释/字符串里提到该名字的文件
    （例如本测试自身）都不会被误判成消费者。
    """
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == symbol for alias in node.names
        ):
            return True
    return False


def _top_level(module: str) -> str:
    return module.split(".", 1)[0]


def test_taiji_substrate_never_imports_legacy_or_transformers() -> None:
    """递归覆盖 taiji/，含未来新增的子包。"""
    offenders: dict[str, set[str]] = {}
    for path in _iter_python_files("taiji"):
        forbidden = {
            module
            for module in _imported_modules(path)
            if _top_level(module) in {"seed", "neuroplex", "transformers"}
        }
        if forbidden:
            offenders[path.relative_to(REPO).as_posix()] = forbidden

    assert not offenders, (
        "Taiji 是 Seed 的自足基底，不得依赖 Seed 上层、Legacy NeuroPlex "
        f"或 HuggingFace transformers：{offenders}"
    )


def test_seed_only_composes_the_public_taiji_substrate() -> None:
    offenders: dict[str, set[str]] = {}
    taiji_imports = 0
    for path in _iter_python_files("seed"):
        modules = _imported_modules(path)
        taiji_imports += sum(_top_level(module) == "taiji" for module in modules)
        forbidden = {
            module
            for module in modules
            if _top_level(module) in {"neuroplex", "transformers"}
        }
        if forbidden:
            offenders[path.relative_to(REPO).as_posix()] = forbidden

    assert taiji_imports > 0, "Seed 必须实际组合 Taiji，而不是只在文档中声明。"
    assert not offenders, f"Seed 不得反向接入 Legacy/Transformer：{offenders}"


def test_legacy_baseline_never_depends_on_seed_or_the_new_substrate() -> None:
    """替代关系必须是单向的：冻结基线不得反向依赖新基底。

    若 neuroplex/ 开始 import taiji，两者就重新耦合，"冻结基线"这一
    参照系随即失效，届时无法再判断回归是新基底还是旧基线引入的。
    """
    offenders: dict[str, set[str]] = {}
    for path in _iter_python_files("neuroplex"):
        forbidden = {
            module
            for module in _imported_modules(path)
            if _top_level(module) in {"seed", "taiji"}
        }
        if forbidden:
            offenders[path.relative_to(REPO).as_posix()] = forbidden

    assert not offenders, (
        "neuroplex/ 是冻结的 Transformer 基线，不得 import seed/taiji（替代关系是单向的）。"
        f"注意 `from taiji.<legacy>` 是 neuroplex 的历史包名别名，不是新基底：{offenders}"
    )


def test_legacy_transformer_block_consumers_stay_closed() -> None:
    """被替代的 Transformer 底层不得获得新的消费者。

    Taiji 顶掉的是 neuroplex/layers.py::TransformerBlock。若这份名单增长，
    说明有人在被替代的底层上继续投入 —— 那是方向回退，必须显式决策。
    """
    consumers: set[str] = set()
    for package in ("neuroplex", "api", "taiji", "tests", "scripts/training"):
        for path in _iter_python_files(package):
            if _imports_symbol(path, "TransformerBlock"):
                consumers.add(path.relative_to(REPO).as_posix())

    assert consumers == LEGACY_TRANSFORMER_CONSUMERS, (
        "Transformer 底层的 live 消费点发生变化。新增消费者请先在 "
        "plans/active/ARCHITECTURE_DIRECTION_2026_08.md 记录理由，"
        f"再更新本测试的名单。实际={sorted(consumers)}"
    )


def test_new_substrate_does_not_reuse_the_legacy_chinese_name() -> None:
    """"态极"是 Legacy NeuroPlex 的旧称，冻结代码内保留但新基底禁止使用。"""
    offenders = [
        path.relative_to(REPO).as_posix()
        for path in _iter_python_files("taiji")
        if "态极" in _read(path)
    ]

    assert not offenders, (
        "\"态极\"指 Legacy NeuroPlex，与替代 Transformer 的新基底无关；"
        f"taiji/ 内请写 \"Taiji\" 或 \"NeuroPlex\"：{offenders}"
    )
