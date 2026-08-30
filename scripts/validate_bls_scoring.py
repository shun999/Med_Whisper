#!/usr/bin/env python3
"""Create annotation sheets and validate BLS scorer outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bls_scoring import DEFAULT_RUBRIC_PATH  # noqa: E402
from src.bls_validation import create_annotation_template, validate_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BLS採点器の研究用検証ツール")
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template", help="専門家・非専門家用の注釈CSVを作成")
    template.add_argument("manifest", type=Path)
    template.add_argument("output", type=Path)
    template.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC_PATH)

    metrics = subparsers.add_parser("metrics", help="注釈と採点JSONから精度指標を計算")
    metrics.add_argument("annotations", type=Path)
    metrics.add_argument("predictions", type=Path, nargs="+")
    metrics.add_argument("--output", type=Path, required=True)
    metrics.add_argument("--bootstrap-samples", type=int, default=1000)
    metrics.add_argument("--seed", type=int, default=20260830)
    metrics.add_argument("--split", default="test", help="集計するsplit（既定: test）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "template":
        count = create_annotation_template(args.manifest, args.output, rubric_path=args.rubric)
        print(f"注釈CSVを作成しました: {args.output} ({count}行)")
        return 0

    report = validate_predictions(
        args.annotations,
        args.predictions,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"検証結果を保存しました: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
