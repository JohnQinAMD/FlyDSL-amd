# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-only provenance checks for the promoted V4 sparse-decode M16 variant."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "kernels" / "attention" / "mla_v4_sparse_decode.py"
COMMON = ROOT / "kernels" / "attention" / "_mla_v4_sparse_decode_common.py"
KERNEL = ROOT / "kernels" / "attention" / "_mla_v4_sparse_decode_kernel.py"

ARTIFACT_SOURCES = (COMMON, KERNEL)

# SHA-256 over the promoted M16/S536 kernel artifact files, sorted and
# concatenated.  Update after intentionally changing shared constants or the
# kernel body and rerunning the validated measurement/correctness checks.
MEASURED_ARTIFACT_SHA256 = "816c371de22efb33a7dfc9d35988d23be64eda4db07694ceb39d5b4a4228c049"


def _artifact_sha256() -> str:
    digest = hashlib.sha256()
    for source in sorted(ARTIFACT_SOURCES):
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


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
    """Tie timing measurements to a specific byte-for-byte artifact."""
    artifact_sha = _artifact_sha256()
    assert artifact_sha == MEASURED_ARTIFACT_SHA256, (
        "M16/S536 common+kernel artifact changed; update "
        f"MEASURED_ARTIFACT_SHA256 to {artifact_sha!r} after review and measurement"
    )


def test_m16_s536_registration_contract_is_present_in_snapshot():
    public_tree = _tree(PUBLIC)
    common_tree = _tree(COMMON)
    assert _literal_assignment(public_tree, "FORGE_VARIANT") == "m16s536"
    assert _literal_assignment(common_tree, "BLOCK_N") == 128
    assert _literal_assignment(common_tree, "NUM_THREADS") == 512
    assert _literal_assignment(common_tree, "OCCUPANCY") == 1
    assert _literal_assignment(common_tree, "M16S536_KV_LDS_STRIDE") == 536
    assert _literal_assignment(common_tree, "M16S536_DYNAMIC_LDS_BYTES") == 142_848

    contract = _literal_assignment(public_tree, "FORGE_REGISTRATION_CONTRACT")
    assert (
        "m16s536",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride",
        (512, 1, 1),
    ) in contract


def test_m16_snapshot_exports_the_public_and_selected_launchers():
    tree = _tree(PUBLIC)
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "flydsl_mla_v4_sparse_decode" in functions
    assert "launch_mla_v4_sparse_decode_n128_token4_m16s536_lds_stride_diagnostic" in functions

    exports = _literal_assignment(tree, "__all__")
    assert "flydsl_mla_v4_sparse_decode" in exports
    assert "launch_mla_v4_sparse_decode_n128_token4_m16s536_lds_stride_diagnostic" in exports
