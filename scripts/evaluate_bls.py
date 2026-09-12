"""Run from the project root: uv run python scripts/evaluate_bls.py --help."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bls_pipeline import (  # noqa: E402
    BLSPipeline, DEVICE_REFERENCE, EVALUATION_MODEL, EVALUATION_RPD, EVALUATION_RPM, TRANSCRIBE_MODEL,
    TRANSCRIPTION_RPD, TRANSCRIPTION_RPM, MIME_TYPES, VIDEO_TYPES, read_json,
)
from src.bls_summary import HUMAN_SCORES_CSV  # noqa: E402


def pipeline_arguments(parser):
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--evaluation-model", default=EVALUATION_MODEL)
    parser.add_argument("--transcription-model", default=TRANSCRIBE_MODEL)
    parser.add_argument("--vocabulary", choices=["baseline", "revised"], default="revised")
    parser.add_argument("--device-reference", type=Path, default=DEVICE_REFERENCE,
                        help="機器音声の参照TXT（既定: data/LED音声人間文字起こし.txt）")
    parser.add_argument("--human-scores-csv", type=Path, default=HUMAN_SCORES_CSV,
                        help="スコアまとめの比較用CSV（既定: data/人間音声採点データ.csv、採点モデルには渡しません）")
    parser.add_argument("--quota-scope", default="default", help="同じGoogleプロジェクトでは同じ値を使用")
    limits = {
        "transcription": (TRANSCRIPTION_RPM, TRANSCRIPTION_RPD),
        "evaluation": (EVALUATION_RPM, EVALUATION_RPD),
    }
    for stage, (rpm, rpd) in limits.items():
        parser.add_argument(f"--{stage}-rpm", type=int, default=rpm)
        parser.add_argument(f"--{stage}-rpd", type=int, default=rpd)


def make_pipeline(args):
    names = ("output_dir", "evaluation_model", "transcription_model", "vocabulary", "quota_scope",
             "transcription_rpm", "transcription_rpd", "evaluation_rpm", "evaluation_rpd", "device_reference", "human_scores_csv")
    return BLSPipeline(**{name: getattr(args, name) for name in names})


def main(argv=None):
    parser = argparse.ArgumentParser(description="BLSの18項目を根拠付きで自動評価します")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--transcript", type=Path, help="既存のUTF-8 TXT（文字起こしAPI不要）")
    inputs.add_argument("--audio", type=Path, help="単一の音声または動画")
    inputs.add_argument("--audio-dir", type=Path, help="音声/動画ディレクトリ（直下のみ）")
    parser.add_argument("--transcribe-only", action="store_true", help="採点APIを呼ばず文字起こしだけ保存")
    parser.add_argument("--dry-run", action="store_true", help="入力と最大リクエスト数を表示（通信・保存なし）")
    pipeline_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if args.audio_dir:
            if not args.audio_dir.is_dir():
                raise ValueError("音声ディレクトリが存在しません")
            paths = sorted(p for p in args.audio_dir.iterdir() if p.is_file() and p.suffix.lower() in MIME_TYPES.keys() | VIDEO_TYPES)
        else:
            paths = [args.transcript or args.audio]
        if not paths or any(not p.is_file() for p in paths):
            raise ValueError("対象ファイルがありません")
        if args.transcript and (args.transcript.suffix.lower() != ".txt" or args.transcribe_only):
            raise ValueError("--transcriptにはTXTを指定し、--transcribe-onlyとは併用しないでください")
        if args.audio and args.audio.suffix.lower() not in MIME_TYPES.keys() | VIDEO_TYPES:
            raise ValueError("--audioには対応する音声または動画を指定してください")
        pipeline = make_pipeline(args)
        if args.dry_run:
            print(f"対象: {len(paths)}本 / 文字起こし最大: {0 if args.transcript else len(paths)}回 / 採点最大: {0 if args.transcribe_only else len(paths)}回（再試行を除く。キャッシュで減少）")
            for path in paths:
                print(path)
            return 0
        run = pipeline.run(paths, transcription_only=args.transcribe_only)
        for entry in run["results"]:
            result = read_json(Path(entry["result_path"]))
            if "evaluation" in result:
                ev = result["evaluation"]
                print(f"{entry['sample_id']}: {ev['score']:.1f}点 ({ev['met_count']}/18、判定不能{ev['uncertain_count']}項目)")
            else:
                print(f"{entry['sample_id']}: 文字起こし保存済み")
        print(f"実行記録: {run['run_path']} ({run['status']})")
        if "summary_path" in run:
            print(f"スコアまとめ: {run['summary_path']}")
        for error in run["errors"]:
            print(f"{error['source_path']}: {error['type']}: {error['message']}", file=sys.stderr)
        return 0 if run["status"] == "completed" else 2
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
