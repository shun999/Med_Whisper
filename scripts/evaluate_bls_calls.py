#!/usr/bin/env python3
"""Evaluate BLS calls in an existing Japanese transcript."""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bls_scoring import (  # noqa: E402
    DEFAULT_RUBRIC_PATH,
    EvaluationResult,
    evaluate_llm,
    evaluate_rules,
    load_rubric,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="文字起こしテキストからBLSコールの内容と順序を採点します。"
    )
    parser.add_argument("transcript", type=Path, help="UTF-8の文字起こしテキスト")
    parser.add_argument(
        "--method",
        choices=("rules", "llm", "both"),
        default="both",
        help="採点方式（既定: both）",
    )
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC_PATH)
    parser.add_argument("--llm-model", default="gemini-3.7-flash")
    parser.add_argument("--llm-repeats", type=int, default=3)
    parser.add_argument(
        "--sample-id",
        default=None,
        help="文字起こしを一意に表すID（既定: 入力ファイル名）",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="同一演習の別カメラ・別マイクで共有するID（既定: sample-id）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation",
    )
    return parser.parse_args()


def _create_gemini_client() -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genaiがありません。`uv sync`を実行してください。") from exc

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key and sys.stdin.isatty():
        api_key = getpass.getpass("Gemini API key: ").strip()
    if not api_key:
        raise RuntimeError(
            "LLM採点にはGEMINI_API_KEYが必要です。環境変数へ設定するか、対話端末で実行してください。"
        )
    return genai.Client(api_key=api_key)


def _write_csv(path: Path, results: list[EvaluationResult]) -> None:
    fields = [
        "record_type",
        "method",
        "id",
        "label",
        "score",
        "status",
        "evidence",
        "char_start",
        "char_end",
        "detected_count",
        "reason",
        "confidence",
        "disagreement",
        "score_stddev",
        "run_agreement",
        "content_score",
        "sequence_score",
        "sequence_coverage",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "record_type": "summary",
                    "method": result.method,
                    "content_score": result.content_score,
                    "sequence_score": result.sequence_score,
                    "sequence_coverage": result.sequence_coverage,
                }
            )
            for call in result.calls:
                writer.writerow(
                    {
                        "record_type": "call",
                        "method": result.method,
                        "id": call.call_id,
                        "label": call.label,
                        "score": call.score,
                        "status": call.status,
                        "evidence": " | ".join(call.evidence),
                        "char_start": call.char_start,
                        "char_end": call.char_end,
                        "detected_count": call.detected_count,
                        "reason": call.reason,
                        "confidence": call.confidence,
                        "disagreement": call.disagreement,
                        "score_stddev": call.score_stddev,
                        "run_agreement": call.run_agreement,
                    }
                )
            for sequence in result.sequences:
                sequence_value = (
                    1.0 if sequence.status == "pass" else 0.0 if sequence.status == "fail" else ""
                )
                writer.writerow(
                    {
                        "record_type": "sequence",
                        "method": result.method,
                        "id": sequence.constraint_id,
                        "label": sequence.label,
                        "score": sequence_value,
                        "status": sequence.status,
                        "char_start": sequence.before_position,
                        "char_end": sequence.after_position,
                        "reason": sequence.reason,
                    }
                )


def main() -> int:
    args = parse_args()
    transcript_path = args.transcript.resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(f"文字起こしが見つかりません: {transcript_path}")
    if args.llm_repeats < 1:
        raise ValueError("--llm-repeatsは1以上にしてください。")

    text = transcript_path.read_text(encoding="utf-8").strip()
    sample_id = args.sample_id or transcript_path.stem
    session_id = args.session_id or sample_id
    rubric = load_rubric(args.rubric)
    results: list[EvaluationResult] = []
    if args.method in {"rules", "both"}:
        results.append(evaluate_rules(text, rubric))
    if args.method in {"llm", "both"}:
        client = _create_gemini_client()
        results.append(
            evaluate_llm(
                text,
                client,
                rubric=rubric,
                model=args.llm_model,
                repeats=args.llm_repeats,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / f"{transcript_path.stem}_evaluation"
    json_path = output_base.with_suffix(".json")
    csv_path = output_base.with_suffix(".csv")
    bundle = {
        "sample_id": sample_id,
        "session_id": session_id,
        "source": str(transcript_path),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "rubric_path": str(args.rubric.resolve()),
        "results": [result.to_dict() for result in results],
    }
    json_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, results)

    for result in results:
        sequence = "評価不能" if result.sequence_score is None else f"{result.sequence_score:.2f}"
        print(
            f"{result.method}: 内容スコア={result.content_score:.2f}, "
            f"順序スコア={sequence}, 順序評価数={result.sequence_coverage}"
        )
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
