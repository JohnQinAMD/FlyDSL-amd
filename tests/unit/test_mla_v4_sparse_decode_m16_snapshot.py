# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-only provenance checks for the measured V4 sparse-decode M16 snapshot."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "kernels" / "attention" / "mla_v4_sparse_decode.py"
MEASURED_M16_SOURCE_SHA256 = "a1355ca0264d8e4556612998b083817439a30755683aad87acf89f68d358958f"


def _tree() -> ast.Module:
    return ast.parse(KERNEL.read_text(encoding="utf-8"), filename=str(KERNEL))


def _literal_assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name and node.value is not None:
                return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment {name!r}")


def test_m16_source_matches_the_authenticated_candidate_snapshot():
    assert hashlib.sha256(KERNEL.read_bytes()).hexdigest() == MEASURED_M16_SOURCE_SHA256


def test_m16_s536_registration_contract_is_present_in_snapshot():
    tree = _tree()
    assert _literal_assignment(tree, "FORGE_VARIANT") == "m16s536"
    assert _literal_assignment(tree, "M16S536_KV_LDS_STRIDE") == 536
    assert _literal_assignment(tree, "M16S536_DYNAMIC_LDS_BYTES") == 142_848

    contract = _literal_assignment(tree, "FORGE_REGISTRATION_CONTRACT")
    assert (
        "m16s536",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride",
        (512, 1, 1),
    ) in contract


def test_m16_snapshot_exports_the_public_and_selected_launchers():
    tree = _tree()
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "flydsl_mla_v4_sparse_decode" in functions
    assert "launch_mla_v4_sparse_decode_n128_token4_m16s536_lds_stride_diagnostic" in functions

    exports = _literal_assignment(tree, "__all__")
    assert "flydsl_mla_v4_sparse_decode" in exports
    assert "launch_mla_v4_sparse_decode_n128_token4_m16s536_lds_stride_diagnostic" in exports
