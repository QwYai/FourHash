#!/usr/bin/env python3
"""Render one trace-backed CCDE shell-refinement case for the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageOps


EVIDENCE_SCHEMA = "raw_rebuilt_visual_evidence_v1"
ANALYSIS_SCHEMA = "raw_rebuilt_ccde_visual_case_analysis_v1"


class RenderError(RuntimeError):
    """Trace-backed evidence does not match the frozen case analysis."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RenderError(f"{path} must contain a JSON object")
    return value


def _active_names(row: Mapping[str, Any]) -> list[str]:
    labels = row.get("label_hot")
    metadata = row.get("metadata")
    if not isinstance(labels, list) or not isinstance(metadata, dict):
        raise RenderError("evidence row lacks labels or metadata")
    names = metadata.get("category_names")
    if not isinstance(names, list) or len(names) != len(labels):
        return [f"label {index}" for index, value in enumerate(labels) if value]
    return [str(names[index]) for index, value in enumerate(labels) if value]


def _query_caption(row: Mapping[str, Any]) -> str:
    raw_text = row.get("raw_text")
    if isinstance(raw_text, dict):
        captions = raw_text.get("captions")
        if isinstance(captions, list) and captions:
            return str(captions[0])
    return str(raw_text)


def _verified_image(case_root: Path, row: Mapping[str, Any]) -> Image.Image:
    raw = row.get("raw_image")
    if not isinstance(raw, dict):
        raise RenderError("evidence row lacks copied raw image")
    path = case_root / str(raw.get("copied_path", ""))
    if (
        not path.is_file()
        or path.stat().st_size != int(raw.get("bytes", -1))
        or _sha256_file(path) != raw.get("sha256")
    ):
        raise RenderError(f"copied image differs from sealed evidence: {path}")
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def _letterbox(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    contained = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, color=(247, 247, 247))
    left = (size[0] - contained.width) // 2
    top = (size[1] - contained.height) // 2
    canvas.paste(contained, (left, top))
    return np.asarray(canvas)


def _verify_inputs(
    evidence_root: Path, analysis_path: Path, case_id: str
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    manifest = _load_json(evidence_root / "evidence_manifest.json")
    analysis = _load_json(analysis_path)
    if manifest.get("schema") != EVIDENCE_SCHEMA:
        raise RenderError("unexpected visual-evidence schema")
    manifest_body = {
        key: manifest[key] for key in manifest if key != "evidence_sha256"
    }
    if hashlib.sha256(_canonical_json_bytes(manifest_body)).hexdigest() != manifest.get(
        "evidence_sha256"
    ):
        raise RenderError("visual-evidence manifest hash changed")
    if analysis.get("schema") != ANALYSIS_SCHEMA or analysis.get("status") != (
        "POSTHOC_ILLUSTRATIVE_ONLY"
    ):
        raise RenderError("unexpected visual-case analysis schema/status")
    analysis_body = {
        key: analysis[key] for key in analysis if key != "analysis_sha256"
    }
    if hashlib.sha256(_canonical_json_bytes(analysis_body)).hexdigest() != analysis.get(
        "analysis_sha256"
    ):
        raise RenderError("visual-case analysis hash changed")
    for key in ("dataset", "source_seal_sha256", "selection_sha256"):
        if manifest.get(key) != analysis.get(key):
            raise RenderError(f"evidence and analysis differ on {key}")
    if manifest.get("rank_token_sha256") != analysis.get("rank_plan_sha256"):
        raise RenderError("evidence rank token differs from the analyzed rank plan")
    evidence_cases = [value for value in manifest.get("cases", []) if value.get("case_id") == case_id]
    analysis_cases = [value for value in analysis.get("cases", []) if value.get("case_id") == case_id]
    if len(evidence_cases) != 1 or len(analysis_cases) != 1:
        raise RenderError(f"case {case_id!r} is not unique in evidence/analysis")
    evidence_case, analysis_case = evidence_cases[0], analysis_cases[0]
    if evidence_case["query"]["canonical_row_id"] != analysis_case["query_row_id"]:
        raise RenderError("query identity differs between evidence and analysis")
    evidence_ids = [value["canonical_row_id"] for value in evidence_case["candidates"]]
    if evidence_ids != analysis_case["candidate_row_ids"]:
        raise RenderError("candidate identity/order differs between evidence and analysis")
    return manifest, evidence_case, analysis_case


def _candidate_annotation(record: Mapping[str, Any]) -> str:
    return (
        rf"$d_p$={int(record['primary_distance'])}  "
        rf"$d_s$={int(record['detail_distance'])}  "
        rf"$J$={float(record['jaccard_gain']):.2f}"
    )


def render_case(
    *,
    evidence_root: Path,
    analysis_path: Path,
    case_id: str,
    output_pdf: Path,
    output_png: Path,
) -> None:
    for output in (output_pdf, output_png):
        if output.exists():
            raise RenderError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    _manifest, evidence_case, analysis_case = _verify_inputs(
        evidence_root, analysis_path, case_id
    )
    case_root = evidence_root / case_id
    query = evidence_case["query"]
    query_image = _letterbox(_verified_image(case_root, query), (760, 520))
    evidence_by_id = {
        value["canonical_row_id"]: value for value in evidence_case["candidates"]
    }
    analysis_records = analysis_case["candidates"]
    if len(analysis_records) != 6:
        raise RenderError("paper layout requires exactly three favored and three demoted rows")
    favored = [value for value in analysis_records if value.get("group") == "favored"]
    demoted = [value for value in analysis_records if value.get("group") == "demoted"]
    if len(favored) != 3 or len(demoted) != 3:
        raise RenderError("paper layout requires a 3+3 strict shell comparison")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.08, 4.70), constrained_layout=False)
    grid = fig.add_gridspec(
        5,
        6,
        height_ratios=(1.05, 0.15, 1.0, 0.15, 1.0),
        left=0.015,
        right=0.992,
        bottom=0.025,
        top=0.985,
        hspace=0.24,
        wspace=0.15,
    )

    query_ax = fig.add_subplot(grid[0, :2])
    query_ax.imshow(query_image)
    query_ax.set_axis_off()
    query_ax.set_title(
        "Source image (not used by T2I query)", loc="left", pad=2, fontsize=8.2
    )
    query_ax.add_patch(
        Rectangle((0, 0), 1, 1, transform=query_ax.transAxes, fill=False, lw=1.0, ec="#555555")
    )

    text_ax = fig.add_subplot(grid[0, 2:])
    text_ax.set_axis_off()
    caption = _query_caption(query)
    labels = ", ".join(_active_names(query))
    text_ax.text(0.0, 0.96, "Text query", weight="bold", va="top", fontsize=9.2)
    text_ax.text(
        0.0,
        0.72,
        "“" + "\n".join(textwrap.wrap(caption, width=68)) + "”",
        va="top",
        fontsize=8.8,
    )
    text_ax.text(
        0.0,
        0.24,
        "Ground-truth concepts: " + labels,
        va="top",
        color="#333333",
        fontsize=7.7,
    )
    text_ax.text(
        0.0,
        0.01,
        (
            f"All six candidates lie in primary shell $d_p$={analysis_case['boundary_primary_distance']}; "
            "only the semantic-code distance $d_s$ is allowed to refine them."
        ),
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )

    colors = {"favored": "#2F6B9A", "demoted": "#C77724"}
    group_labels = {
        "favored": "Lower semantic-code distance — favored inside the shell",
        "demoted": "Higher semantic-code distance — demoted inside the shell",
    }
    for group_index, (group, records) in enumerate((("favored", favored), ("demoted", demoted))):
        header_row = 1 + 2 * group_index
        candidate_row = header_row + 1
        header_ax = fig.add_subplot(grid[header_row, :])
        header_ax.set_axis_off()
        header_ax.text(
            0.0,
            0.25,
            group_labels[group],
            color=colors[group],
            weight="bold",
            fontsize=8.1,
            va="center",
        )
        for column_group, record in enumerate(records):
            ax = fig.add_subplot(
                grid[candidate_row, 2 * column_group : 2 * column_group + 2]
            )
            evidence = evidence_by_id[str(record["row_id"])]
            image = _letterbox(_verified_image(case_root, evidence), (700, 510))
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(colors[group])
                spine.set_linewidth(1.5)
            active = ", ".join(_active_names(evidence)) or "no shared class"
            if len(active) > 36:
                active = active[:33].rstrip() + "…"
            ax.text(
                0.5,
                0.985,
                _candidate_annotation(record) + "\n" + active,
                transform=ax.transAxes,
                ha="center",
                va="top",
                color="#222222",
                fontsize=7.1,
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
            )

    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    render_case(
        evidence_root=args.evidence.expanduser().resolve(strict=True),
        analysis_path=args.analysis.expanduser().resolve(strict=True),
        case_id=args.case_id,
        output_pdf=args.output_pdf.expanduser().resolve(),
        output_png=args.output_png.expanduser().resolve(),
    )
    print(json.dumps({"pdf": str(args.output_pdf), "png": str(args.output_png)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
