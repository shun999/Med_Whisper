import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_bls import make_pipeline, pipeline_arguments  # noqa: E402
from src.bls_benchmark import init_manifest, report, run_references  # noqa: E402
from src.bls_pipeline import atomic_text, read_json, safe_output, write_json  # noqa: E402
from src.bls_summary import HUMAN_SCORES_CSV, load_human_scores, render_score_summary  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="BLS正解ラベルの準備と精度検証（未注釈を補完しません）")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="音声を変更せず正解ラベルの雛形を作成")
    init.add_argument("--audio-dir", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("report", help="ASR採点と正解を比較")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--run", type=Path, required=True)
    compare.add_argument("--reference-run", type=Path)
    compare.add_argument("--split", choices=["development", "validation", "all"], default="validation")
    compare.add_argument("--output", type=Path, required=True)
    references = commands.add_parser("references", help="人が確認した逐語録を同じ評価器で採点")
    references.add_argument("--manifest", type=Path, required=True)
    references.add_argument("--output", type=Path, required=True)
    pipeline_arguments(references)
    summary = commands.add_parser("summary", help="保存済み採点と人手CSVからスコアまとめTXTを作成（API不要）")
    summary.add_argument("--run", type=Path, required=True)
    summary.add_argument("--human-scores-csv", type=Path, default=HUMAN_SCORES_CSV)
    summary.add_argument("--results-dir", type=Path, help="結果JSONの移動先（省略時は元の保存先の配下を検索）")
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = init_manifest(args.audio_dir, args.output)
            print(f"{len(result['samples'])}本の正解ラベル雛形: {args.output}")
        elif args.command == "references":
            result = run_references(read_json(args.manifest), make_pipeline(args), args.output)
            print(f"{result['status']}: {args.output}")
            return 0 if result["status"] == "completed" else 2
        elif args.command == "summary":
            destination = safe_output(args.output)
            if destination.suffix.lower() != ".txt":
                raise ValueError("スコアまとめの保存先にはTXTを指定してください")
            if destination.exists():
                raise FileExistsError("既存のスコアまとめは上書きしません。新しい保存先を指定してください")
            run = read_json(args.run)
            run["run_path"] = str(args.run.resolve())
            summary_text = render_score_summary(run, load_human_scores(args.human_scores_csv), results_dir=args.results_dir)
            atomic_text(destination, summary_text)
            print(f"スコアまとめ: {destination}")
        else:
            if args.output.exists():
                raise FileExistsError("既存レポートは上書きしません")
            result = report(read_json(args.manifest), read_json(args.run),
                            reference_run=read_json(args.reference_run) if args.reference_run else None, split=args.split)
            write_json(args.output, result)
            print(json.dumps({k: result[k] for k in ("evaluated_labeled_items", "micro", "score_mae", "uncertain_rate", "cer")}, ensure_ascii=False, indent=2))
            print(f"詳細: {args.output}")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
