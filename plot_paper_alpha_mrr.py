"""Plot MRR vs alpha from the paper evaluation markdown report.

The script writes a self-contained SVG so it can run in the project
environment without adding plotting dependencies. Rows with an alpha value are
drawn as alpha curves. Rows without an alpha value, such as temporal-only and
semantic-only baselines, are drawn as constant horizontal lines across the
alpha range.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.sax.saxutils import escape

from temporal_embeddings.utils.os.folder_management import create_folders


DEFAULT_REPORT = Path("output/metrics/paper_evaluation_report.md")
DEFAULT_OUTPUT = Path("output/metrics/paper_alpha_mrr.svg")
DEFAULT_ALPHAS = [round(0.1 * i, 1) for i in range(1, 10)]

BENCHMARK_DISPLAY: Dict[str, str] = {
    "time_sensitive_qa": "TimeQA",
    "temp_reason": "TempReason",
    "temp_reason_l1": "TempReason L1",
    "temp_reason_l2": "TempReason L2",
    "ts_retriever": "TS-Retriever",
}

SEMANTIC_DISPLAY: Dict[str, str] = {
    "all-mpnet-base-v2": "MPNet-base-v2",
    "intfloat/e5-base-v2": "E5-base-v2",
    "BAAI/bge-large-en-v1.5": "BGE-large-v1.5",
    "salesforce": "Mistral",
}

MODEL_COLORS: Dict[str, str] = {
    "all-minilm-l6-v2": "#4c4c4c",
    "all-minilm-l6-v2-full": "#4c4c4c",
    "all-mpnet-base-v2": "#1f77b4",
    "intfloat/e5-base-v2": "#ff7f0e",
    "BAAI/bge-large-en-v1.5": "#2ca02c",
    "salesforce": "#d62728",
}

TYPE_ORDER: Dict[str, int] = {
    "temporal": 0,
    "semantic": 1,
    "hybrid": 2,
    "sutime_hybrid": 3,
}

TYPE_STYLES: Dict[str, Dict[str, str]] = {
    "temporal": {"dash": "7 5", "marker": ""},
    "semantic": {"dash": "2 5", "marker": ""},
    "hybrid": {"dash": "", "marker": "circle"},
    "sutime_hybrid": {"dash": "8 4 2 4", "marker": "square"},
}


@dataclass(frozen=True)
class PaperResult:
    run_id: str
    run_type: str
    temporal_model_name: str
    semantic_model_name: str
    alpha: Optional[float]
    benchmark: str
    num_queries: int
    mrr: float


def parse_paper_report(report_path: Path) -> List[PaperResult]:
    rows: List[PaperResult] = []

    with report_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("|") or line.startswith("| ---") or line.startswith("| Run ID"):
                continue

            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 11:
                continue

            (
                run_id,
                run_type,
                temporal_model_name,
                semantic_model_name,
                alpha_value,
                benchmark,
                num_queries,
                mrr,
                _ndcg,
                _recall,
                _precision,
            ) = cells

            alpha = None if alpha_value in ("", "-") else float(alpha_value)
            rows.append(
                PaperResult(
                    run_id=run_id,
                    run_type=run_type,
                    temporal_model_name=temporal_model_name,
                    semantic_model_name=semantic_model_name,
                    alpha=alpha,
                    benchmark=benchmark,
                    num_queries=int(num_queries),
                    mrr=float(mrr),
                )
            )

    if not rows:
        raise ValueError(f"No paper-result rows found in {report_path}")

    return rows


def display_benchmark(benchmark: str) -> str:
    return BENCHMARK_DISPLAY.get(benchmark, benchmark)


def display_semantic_model(model_name: str) -> str:
    return SEMANTIC_DISPLAY.get(model_name, model_name)


def display_run(row: PaperResult) -> str:
    if row.run_type == "temporal":
        if row.run_id == "temporal:all-minilm-l6-v2":
            return "TEMPS w/o Semantic Model"
        return row.temporal_model_name or row.run_id
    if row.run_type == "semantic":
        return display_semantic_model(row.semantic_model_name)
    if row.run_type == "hybrid":
        return f"{display_semantic_model(row.semantic_model_name)} + TEMPS"
    if row.run_type == "sutime_hybrid":
        return f"{display_semantic_model(row.semantic_model_name)} + SUTime"
    return row.run_id


def run_color(row: PaperResult) -> str:
    if row.semantic_model_name:
        return MODEL_COLORS.get(row.semantic_model_name, "#333333")
    return MODEL_COLORS.get(row.temporal_model_name, "#333333")


def run_sort_key(row: PaperResult) -> Tuple[int, str]:
    return TYPE_ORDER.get(row.run_type, 99), display_run(row)


def benchmark_order(rows: Iterable[PaperResult]) -> List[str]:
    available = {row.benchmark for row in rows}
    ordered = [benchmark for benchmark in BENCHMARK_DISPLAY if benchmark in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def alpha_range(rows: Iterable[PaperResult]) -> List[float]:
    alphas = sorted({row.alpha for row in rows if row.alpha is not None})
    return [float(alpha) for alpha in alphas] or DEFAULT_ALPHAS


def filter_rows(
    rows: List[PaperResult],
    benchmarks: Optional[List[str]],
    run_types: Optional[List[str]],
    semantic_models: Optional[List[str]],
) -> List[PaperResult]:
    benchmark_filter = set(benchmarks) if benchmarks else None
    run_type_filter = set(run_types) if run_types else None
    semantic_filter = set(semantic_models) if semantic_models else None

    filtered: List[PaperResult] = []
    for row in rows:
        if benchmark_filter is not None and row.benchmark not in benchmark_filter:
            continue
        if run_type_filter is not None and row.run_type not in run_type_filter:
            continue
        if semantic_filter is not None and row.semantic_model_name:
            if row.semantic_model_name not in semantic_filter:
                continue
        filtered.append(row)

    if not filtered:
        raise ValueError("No rows matched the requested filters.")

    return filtered


def grouped_by_run(rows: Iterable[PaperResult]) -> Dict[str, List[PaperResult]]:
    grouped: Dict[str, List[PaperResult]] = {}
    for row in rows:
        grouped.setdefault(row.run_id, []).append(row)
    return grouped


def y_limits(rows: List[PaperResult]) -> Tuple[float, float]:
    values = [row.mrr for row in rows]
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-12:
        return max(0.0, low - 0.05), min(1.0, high + 0.05)

    padding = max((high - low) * 0.08, 0.02)
    return max(0.0, low - padding), min(1.0, high + padding)


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 12,
    anchor: str = "start",
    weight: str = "normal",
    rotate: Optional[float] = None,
) -> str:
    transform = ""
    if rotate is not None:
        transform = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"'
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, Helvetica, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="#222"{transform}>{escape(text)}</text>'
    )


def svg_marker(x: float, y: float, marker: str, color: str) -> str:
    if marker == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" />'
    if marker == "square":
        return f'<rect x="{x - 3.2:.1f}" y="{y - 3.2:.1f}" width="6.4" height="6.4" fill="{color}" />'
    return ""


def svg_polyline(
    points: List[Tuple[float, float]],
    color: str,
    dash: str,
    width: float,
) -> str:
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{point_text}" fill="none" stroke="{color}" '
        f'stroke-width="{width:.1f}" stroke-linecap="round" stroke-linejoin="round"{dash_attr} />'
    )


def write_svg_alpha_mrr(
    rows: List[PaperResult],
    output_path: Path,
    title: str,
) -> None:
    benchmarks = benchmark_order(rows)
    alphas = alpha_range(rows)
    x_min, x_max = min(alphas), max(alphas)
    x_span = max(x_max - x_min, 1e-12)

    cols = 2 if len(benchmarks) > 1 else 1
    subplot_rows = ceil(len(benchmarks) / cols)

    title_height = 48
    outer_margin = 36
    cell_width = 560
    cell_height = 360
    plot_left = 72
    plot_top = 44
    plot_width = 430
    plot_height = 245
    legend_width = 310

    width = outer_margin * 2 + cols * cell_width + legend_width
    height = title_height + outer_margin + subplot_rows * cell_height

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        svg_text(width / 2, 28, title, size=18, anchor="middle", weight="bold"),
    ]

    legend_items: Dict[str, Tuple[str, str, str]] = {}

    for idx, benchmark in enumerate(benchmarks):
        row_idx = idx // cols
        col_idx = idx % cols
        origin_x = outer_margin + col_idx * cell_width
        origin_y = title_height + row_idx * cell_height
        benchmark_rows = [row for row in rows if row.benchmark == benchmark]
        low, high = y_limits(benchmark_rows)
        y_span = max(high - low, 1e-12)

        def sx(alpha: float) -> float:
            return origin_x + plot_left + (alpha - x_min) / x_span * plot_width

        def sy(mrr: float) -> float:
            return origin_y + plot_top + plot_height - (mrr - low) / y_span * plot_height

        plot_x = origin_x + plot_left
        plot_y = origin_y + plot_top
        parts.append(svg_text(origin_x + plot_left, origin_y + 24, display_benchmark(benchmark), size=15, weight="bold"))
        parts.append(
            f'<rect x="{plot_x:.1f}" y="{plot_y:.1f}" width="{plot_width}" height="{plot_height}" '
            'fill="#fbfbfb" stroke="#d6d6d6" stroke-width="1" />'
        )

        for tick_idx in range(5):
            value = low + (high - low) * tick_idx / 4.0
            y = sy(value)
            parts.append(
                f'<line x1="{plot_x:.1f}" x2="{plot_x + plot_width:.1f}" y1="{y:.1f}" y2="{y:.1f}" '
                'stroke="#e6e6e6" stroke-width="1" />'
            )
            parts.append(svg_text(plot_x - 10, y + 4, f"{value:.2f}", size=10, anchor="end"))

        for alpha in alphas:
            x = sx(alpha)
            parts.append(
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{plot_y:.1f}" y2="{plot_y + plot_height:.1f}" '
                'stroke="#eeeeee" stroke-width="1" />'
            )
            parts.append(svg_text(x, plot_y + plot_height + 20, f"{alpha:.1f}", size=10, anchor="middle"))

        parts.append(svg_text(plot_x + plot_width / 2, plot_y + plot_height + 44, "Alpha", size=12, anchor="middle"))
        parts.append(svg_text(plot_x - 52, plot_y + plot_height / 2, "MRR", size=12, anchor="middle", rotate=-90.0))

        grouped = grouped_by_run(benchmark_rows)
        for _run_id, run_rows in sorted(grouped.items(), key=lambda item: run_sort_key(item[1][0])):
            first_row = run_rows[0]
            label = display_run(first_row)
            color = run_color(first_row)
            style = TYPE_STYLES.get(first_row.run_type, TYPE_STYLES["hybrid"])
            dash = style["dash"]
            marker = style["marker"]
            width_px = 2.1 if first_row.run_type in ("hybrid", "temporal", "semantic") else 1.9

            alpha_rows = sorted(
                [row for row in run_rows if row.alpha is not None],
                key=lambda row: row.alpha or 0.0,
            )
            if alpha_rows:
                points = [(sx(row.alpha or x_min), sy(row.mrr)) for row in alpha_rows]
            else:
                points = [(sx(x_min), sy(first_row.mrr)), (sx(x_max), sy(first_row.mrr))]

            parts.append(svg_polyline(points, color=color, dash=dash, width=width_px))
            for x, y in points:
                parts.append(svg_marker(x, y, marker, color))

            legend_items.setdefault(label, (color, dash, marker))

    legend_x = outer_margin + cols * cell_width + 28
    legend_y = title_height + 26
    parts.append(svg_text(legend_x, legend_y, "Runs", size=14, weight="bold"))
    for idx, (label, (color, dash, marker)) in enumerate(legend_items.items()):
        y = legend_y + 28 + idx * 24
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{legend_x:.1f}" x2="{legend_x + 36:.1f}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="2.2" stroke-linecap="round"{dash_attr} />'
        )
        parts.append(svg_marker(legend_x + 18, y, marker, color))
        parts.append(svg_text(legend_x + 48, y + 4, label, size=11))

    parts.append("</svg>")

    create_folders([output_path.parent])
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")

    print(f"Saved figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot MRR vs alpha from a paper evaluation markdown report."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Optional benchmark ids to plot, e.g. time_sensitive_qa ts_retriever.",
    )
    parser.add_argument(
        "--run-types",
        nargs="+",
        choices=["temporal", "semantic", "hybrid", "sutime_hybrid"],
        default=None,
        help="Optional run types to include. Defaults to all run types.",
    )
    parser.add_argument(
        "--semantic-models",
        nargs="+",
        default=None,
        help="Optional semantic model ids to keep, e.g. salesforce intfloat/e5-base-v2.",
    )
    parser.add_argument("--title", default="Paper evaluation MRR by alpha")

    args = parser.parse_args()

    rows = parse_paper_report(args.report)
    rows = filter_rows(
        rows=rows,
        benchmarks=args.benchmarks,
        run_types=args.run_types,
        semantic_models=args.semantic_models,
    )
    write_svg_alpha_mrr(
        rows=rows,
        output_path=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
