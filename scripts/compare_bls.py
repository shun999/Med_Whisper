"""Controlled BLS experiments; gold labels are used only by local reports."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bls_benchmark import report, validate_manifest  # noqa: E402
from src.bls_pipeline import BLSPipeline, atomic_text, read_json, safe_output, write_json  # noqa: E402


STAGES = {
    "counts": dict(evaluation_profile="counts", vocabulary="revised"),
    "improved": dict(evaluation_profile="improved", vocabulary="revised"),
    "vocabulary": dict(evaluation_profile="improved", vocabulary="targeted"),
    "speakers": dict(evaluation_profile="improved", vocabulary="none", transcription_profile="speakers"),
}


def compare_metrics(baseline: dict, candidate: dict) -> dict:
    baseline_ids = {s["sample_id"] for s in baseline["scores"]}
    candidate_ids = {s["sample_id"] for s in candidate["scores"]}
    comparable = (baseline["run_status"] == candidate["run_status"] == "completed" and bool(baseline_ids)
                  and baseline_ids == candidate_ids
                  and baseline["evaluated_labeled_items"] == candidate["evaluated_labeled_items"])
    accepted = (comparable and candidate["micro"]["fp"] <= baseline["micro"]["fp"]
                and candidate["score_mae"] < baseline["score_mae"])
    return {"comparable": comparable, "improves_baseline": accepted,
            "mae_delta": candidate["score_mae"] - baseline["score_mae"] if comparable else None,
            "false_positive_delta": candidate["micro"]["fp"] - baseline["micro"]["fp"] if comparable else None}


def write_comparison(destination: Path, comparison: dict, baseline: dict) -> None:
    candidates = [s for s in comparison["stages"] if s["improves_baseline"]]
    best = min(candidates, key=lambda s: (s["score_mae"], list(STAGES).index(s["stage"]))) if candidates else None
    comparison["best_completed_stage"] = best["stage"] if best else None
    write_json(destination / "comparison.json", comparison)
    lines = ["# BLS採点の段階比較", "", "| 設定 | MAE（点） | 見逃し | 誤加点 | 基準より改善 |",
             "|---|---:|---:|---:|---|",
             f"| revised3 | {baseline['score_mae']:.3f} | {baseline['micro']['fn']} | {baseline['micro']['fp']} | 基準 |"]
    for result in comparison["stages"]:
        if not result["comparable"]:
            lines.append(f"| {result['stage']} | 比較未完了 | — | — | — |")
        else:
            lines.append(f"| {result['stage']} | {result['score_mae']:.3f} | {result['micro']['fn']} | "
                         f"{result['micro']['fp']} | {'はい' if result['improves_baseline'] else 'いいえ'} |")
    lines += ["", "counts: 数唱基準変更。improved: AED操作場面の照合も変更。vocabulary: 語彙追加。speakers: 語彙なしの話者識別・時刻情報。",
              "", f"完了した設定のうち最良: {best['stage'] if best else 'なし'}。陰性の回帰テストも採用条件とする。",
              "", comparison["limitations"], "", "人が確認した逐語録がないため、CERと音声認識由来の誤りの確定評価は未実施。"]
    atomic_text(destination / "comparison.md", "\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="新しい実験ディレクトリ")
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        destination = safe_output(args.output)
        if destination.exists():
            raise FileExistsError("実験ディレクトリは新しい保存先を指定してください（APIキャッシュは共有します）")
        if len(set(args.stages)) != len(args.stages):
            raise ValueError("実験設定が重複しています")
        manifest = read_json(args.manifest)
        samples = validate_manifest(manifest)
        if any(any(v is None for v in s["labels"].values()) for s in samples):
            raise ValueError("比較には全項目の正解ラベルが必要です")
        paths = [Path(s["audio_path"]) for s in samples]
        baseline = report(manifest, read_json(args.baseline_run), split="all", saved_results=True)
        if baseline["evaluated_labeled_items"] != len(samples) * 18:
            raise ValueError("基準runは全音声の採点が必要です")
        print(f"基準: MAE={baseline['score_mae']:.3f}, FN={baseline['micro']['fn']}, FP={baseline['micro']['fp']}", flush=True)
        if args.dry_run:
            print(f"対象{len(paths)}本、設定{args.stages}。設定ごとに文字起こし最大{len(paths)}回・採点最大{len(paths)}回（キャッシュで減少、再試行を除く）。")
            return 0
        write_json(destination / "baseline.json", baseline)
        comparison = {"created_at": datetime.now(timezone.utc).isoformat(), "baseline_run": str(args.baseline_run.resolve()),
                      "manifest": str(args.manifest.resolve()), "stages": [],
                      "limitations": "同一収録日で既に探索済みの20本。独立した汎化評価ではない。規則テストの合格も採用条件。"}
        for name in args.stages:
            print(f"開始: {name}", flush=True)
            pipeline = BLSPipeline(**STAGES[name])
            run = pipeline.run(paths)
            write_json(destination / f"{name}_run.json", run)
            if run["results"]:
                measured = report(manifest, run, split="all")
                write_json(destination / f"{name}_report.json", measured)
                outcome = {"stage": name, "settings": STAGES[name], "run_path": run["run_path"],
                           "status": run["status"], "score_mae": measured["score_mae"], "micro": measured["micro"],
                           **compare_metrics(baseline, measured)}
            else:
                outcome = {"stage": name, "settings": STAGES[name], "run_path": run["run_path"],
                           "status": run["status"], "comparable": False, "improves_baseline": False}
            comparison["stages"].append(outcome)
            write_comparison(destination, comparison, baseline)
            print(json.dumps(outcome, ensure_ascii=False), flush=True)
            if run["status"] != "completed":
                print(json.dumps(run["errors"], ensure_ascii=False), file=sys.stderr)
                return 2
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
