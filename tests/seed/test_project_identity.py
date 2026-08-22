from __future__ import annotations

import tomllib
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_distribution_and_readme_are_seed() -> None:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    assert pyproject["project"]["name"] == "seed"
    assert "seed*" in pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    assert readme.startswith("# Seed —")
    assert "Seed is the project" not in readme  # avoid reintroducing two identities


def test_desktop_build_artifact_is_seed() -> None:
    build = (REPO / "desktop" / "build.py").read_text(encoding="utf-8")
    installer = (REPO / "desktop" / "installer.nsi").read_text(encoding="utf-8")

    assert (REPO / "desktop" / "seed.spec").is_file()
    assert not (REPO / "desktop" / "neuroplex.spec").exists()
    assert '"--name=Seed"' in build
    assert '!define APP_EXE "Seed.exe"' in installer


def test_legacy_neuroplex_is_explicitly_a_frozen_comparison() -> None:
    direction = (REPO / "plans" / "active" / "ARCHITECTURE_DIRECTION_2026_08.md")
    text = direction.read_text(encoding="utf-8")

    assert (REPO / "neuroplex").is_dir()
    assert "**Legacy NeuroPlex**" in text
    assert "冻结的 Transformer 基线" in text
