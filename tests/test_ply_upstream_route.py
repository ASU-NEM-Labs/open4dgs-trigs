"""Static guardrails for the pinned upstream PLY control path."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib


PACKAGE_ROOT = Path(__file__).parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src" / "trigs"


def test_console_script_routes_only_ply_to_upstream_control() -> None:
    project = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())
    scripts = project["project"]["scripts"]

    assert scripts["trigs-train-plys"] == "trigs.cli.train_from_plys_upstream:main"
    assert scripts["trigs-train-colmap"] == "trigs.cli.train_from_colmap:main"
    assert scripts["trigs-train-selfcap"] == "trigs.cli.train_from_selfcap:main"


def test_upstream_ply_cli_imports_isolated_trainer() -> None:
    cli_path = SOURCE_ROOT / "cli" / "train_from_plys_upstream.py"
    cli_source = cli_path.read_text()
    module = ast.parse(cli_source)
    imported_modules = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
    }

    assert "trigs.core.trainer_ply_upstream" in imported_modules
    assert "trigs.core.trainer" not in imported_modules
    assert "bb51ea6482513bb3075a11e959a2beec46f1702b" in cli_source


def test_upstream_trainer_uses_isolated_relocation() -> None:
    trainer = (
        SOURCE_ROOT / "core" / "trainer_ply_upstream.py"
    ).read_text()

    assert "motion_guided_relocation_ply_upstream" in trainer


def test_stale_ply_console_entrypoint_redirects_upstream() -> None:
    legacy_cli = (SOURCE_ROOT / "cli" / "train_from_plys.py").read_text()

    assert "def _main_extended(" in legacy_cli
    assert "from trigs.cli.train_from_plys_upstream import main as upstream_main" in legacy_cli
    assert "upstream_main()" in legacy_cli
