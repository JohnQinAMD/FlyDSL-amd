# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-only provenance and packaging checks for DSV4 sparse-MLA training."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FWD = ROOT / "kernels" / "attention" / "mla_v4_sparse_train_fwd.py"
BWD = ROOT / "kernels" / "attention" / "mla_v4_sparse_train_bwd.py"

# Canonical pre-port KernelForge source snapshots. The standalone files have
# different hashes because their imports, license headers, and environment
# variable namespace were adapted for FlyDSL 0.3.
KERNELFORGE_FWD_SHA256 = "5525643d5343b99a06897a94f67d3aa86586b6a8138e310247fb65336e9f8b3e"
KERNELFORGE_BWD_SHA256 = "6864d3e0863652d69481fc6a7a5e71ee9b04516b5829ddea8df8cc44e01b577c"
SCORE_K_AST_SHA256 = "4edf39c1904257eafd18f797c25a356dca7ef5f95887025fe1445eb583f1378c"
PV_K_DMAJOR_AST_SHA256 = "60042dafed892149d3d5456a33ddee413f53df0f15baacdf34d238cb777e8f7f"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function_node(path: Path, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    source = path.read_text(encoding="utf-8")
    matches = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def _function_ast_sha256(path: Path, name: str) -> str:
    normalized = ast.dump(
        _function_node(path, name),
        annotate_fields=True,
        include_attributes=False,
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


def test_training_snapshot_retains_the_accepted_u11b_kernel_bodies() -> None:
    assert _function_ast_sha256(BWD, "score_k") == SCORE_K_AST_SHA256
    assert _function_ast_sha256(BWD, "pv_k_dmajor") == PV_K_DMAJOR_AST_SHA256


def test_training_snapshot_is_standalone_flydsl_03_source() -> None:
    for path in (FWD, BWD):
        source = path.read_text(encoding="utf-8")
        assert "primus_turbo" not in source
        assert "from flydsl.expr import arith, buffer_ops" not in source
        assert "from kernels.common import buffer_ops" in source
        assert source.startswith("# SPDX-License-Identifier: MIT")

    backward = BWD.read_text(encoding="utf-8")
    assert "def _pv_k_p13_reference(" not in backward
    assert "def pv_k_mfma32(" not in backward
    assert "PRIMUS_TURBO_FLYDSL_DSV4" not in backward


def test_training_snapshot_exports_the_public_entry_points() -> None:
    expected = {
        FWD: ("sparse_mla_fwd_flydsl",),
        BWD: ("sparse_mla_bwd_flydsl",),
    }
    for path, exports in expected.items():
        tree = _tree(path)
        functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == "__all__"
        }
        assert tuple(assignments["__all__"]) == exports
        assert set(exports) <= functions


def test_training_api_requires_explicit_canonical_topk_opt_in() -> None:
    for path, entry_point in (
        (FWD, "sparse_mla_fwd_flydsl"),
        (BWD, "sparse_mla_bwd_flydsl"),
    ):
        function = _function_node(path, entry_point)
        assert [argument.arg for argument in function.args.kwonlyargs] == ["canonical_topk"]
        assert [ast.literal_eval(default) for default in function.args.kw_defaults] == [False]
        source = ast.unparse(function)
        assert "attn_sink is required by the DSV4 training contract" in source
