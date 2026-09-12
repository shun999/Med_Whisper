"""Summary arithmetic and local integration, using synthetic data and no API calls."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.benchmark_bls import main as benchmark_main
from src.bls_pipeline import BLSPipeline
from src.bls_summary import load_human_scores, render_score_summary


class SummaryTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.base = Path(folder.name)
        self.csv = self.base / "模範.csv"

    def write_gold(self, rows):
        with self.csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["BLS評価対象コール", *[f"項目{i}" for i in range(1, 19)], "成功項目数", "点数"])
            writer.writerows(rows)

    def gold_row(self, sid, met_count):
        return [sid, *([1] * met_count + [0] * (18 - met_count)), met_count, met_count / 18 * 100]

    def result(self, sid, met_count, *, uncertain=False):
        path = self.base / "results" / f"{sid}_exact_hash.json"
        path.parent.mkdir(exist_ok=True)
        items = [{"id": i, "status": "met" if i <= met_count else "uncertain" if uncertain else "not_detected"}
                 for i in range(1, 19)]
        record = {"sample_id": sid, "input_sha256": sid, "evaluation_model": "synthetic",
                  "evaluation": {"score": round(met_count / 18 * 100, 1), "met_count": met_count,
                                 "uncertain_count": 18 - met_count if uncertain else 0, "items": items}}
        path.write_text(json.dumps(record), encoding="utf-8")
        return {"sample_id": sid, "input_sha256": sid, "result_path": str(path)}

    def run_record(self, entries, **kwargs):
        return {"status": "completed", "stage": "evaluation", "results": entries, "errors": [], "pending": [], **kwargs}

    def test_bom_rounding_signed_difference_and_natural_order(self):
        self.write_gold([self.gold_row("20260706_1回目.wav", 18), self.gold_row("20260706_2回目", 16),
                         self.gold_row("20260706_10回目", 17)])
        entries = [self.result("20260706_10回目", 17), self.result("20260706_2回目", 17), self.result("20260706_1回目", 17)]
        text = render_score_summary(self.run_record(entries), load_human_scores(self.csv))
        self.assertIn("Gemini=94.4点 / 模範=100.0点 / 差=-5.6点", text)
        self.assertIn("Gemini=94.4点 / 模範=88.9点 / 差=+5.5点", text)
        self.assertIn("平均絶対誤差（MAE）: 3.7点", text)
        self.assertIn("点数一致: 1/3本", text)
        self.assertLess(text.index("20260706_1回目:"), text.index("20260706_2回目:"))
        self.assertLess(text.index("20260706_2回目:"), text.index("20260706_10回目:"))

    def test_unknown_gold_is_excluded_from_mean(self):
        self.write_gold([self.gold_row("sample", 18)])
        entries = [self.result("sample", 17), self.result("unlabeled", 0)]
        text = render_score_summary(self.run_record(entries), load_human_scores(self.csv))
        self.assertIn("採点済み: 2本 / 比較可能: 1本 / 模範なし: 1本", text)
        self.assertIn("模範=なし / 差=比較不可", text)
        self.assertIn("平均絶対誤差（MAE）: 5.6点", text)
        text = render_score_summary(self.run_record(entries), load_human_scores(None))
        self.assertIn("全体の乖離: 比較対象がないため算出不可", text)

    def test_malformed_csv_rejected_including_duplicates_and_inconsistent_scores(self):
        for mode in ("blank", "nan", "count", "score", "columns", "duplicate"):
            with self.subTest(mode=mode):
                row = self.gold_row("sample", 18)
                if mode == "blank":
                    row[1] = ""
                elif mode == "nan":
                    row[-1] = "NaN"
                elif mode == "count":
                    row[-2] = 17
                elif mode == "score":
                    row[-1] = 77.8
                elif mode == "columns":
                    row.pop()
                self.write_gold([row, row] if mode == "duplicate" else [row])
                with self.assertRaises(ValueError):
                    load_human_scores(self.csv)

    def test_same_score_can_hide_item_disagreement_and_uncertainty(self):
        self.write_gold([self.gold_row("sample", 17)])
        entry = self.result("sample", 17)
        path = Path(entry["result_path"])
        record = json.loads(path.read_text())
        record["evaluation"]["items"][0]["status"] = "not_detected"
        record["evaluation"]["items"][17]["status"] = "met"
        path.write_text(json.dumps(record), encoding="utf-8")
        text = render_score_summary(self.run_record([entry]), load_human_scores(self.csv))
        self.assertIn("点数一致: 1/1本", text)
        self.assertIn("1.傷病者発見（模範:達成 / Gemini:未検出）", text)
        self.assertIn("18.荷物の存在を知らせる（模範:未達成 / Gemini:達成）", text)

    def test_moved_result_lookup_is_exact_and_rejects_ambiguity(self):
        self.write_gold([self.gold_row("sample", 18)])
        entry = self.result("sample", 18)
        old = Path(entry["result_path"])
        moved_dir = old.parent / "revised3"
        moved_dir.mkdir()
        moved = moved_dir / old.name
        old.rename(moved)
        text = render_score_summary(self.run_record([entry]), load_human_scores(self.csv))
        self.assertIn("点数一致: 1/1本", text)
        other = old.parent / "duplicate"
        other.mkdir()
        (other / old.name).write_bytes(moved.read_bytes())
        with self.assertRaises(ValueError):
            render_score_summary(self.run_record([entry]), load_human_scores(self.csv))
        text = render_score_summary(self.run_record([entry]), load_human_scores(self.csv), results_dir=moved_dir)
        self.assertIn("点数一致: 1/1本", text)

    def test_interrupted_no_results_is_not_zero_error(self):
        self.write_gold([self.gold_row("sample", 18)])
        run = self.run_record([], status="interrupted", pending=["sample.wav", "later.wav"],
                              errors=[{"source_path": "sample.wav", "type": "QuotaExhausted"}])
        text = render_score_summary(run, load_human_scores(self.csv))
        self.assertIn("比較対象がないため算出不可", text)
        self.assertIn("sample: エラー: QuotaExhausted", text)
        self.assertIn("later: 未処理", text)
        self.assertNotIn("Gemini=0.0", text)

    def test_summary_cli_uses_only_results_and_does_not_overwrite(self):
        self.write_gold([self.gold_row("sample", 18)])
        run_path = self.base / "run.json"
        run_path.write_text(json.dumps(self.run_record([self.result("sample", 18)])), encoding="utf-8")
        output = self.base / "scores.txt"
        args = ["summary", "--run", str(run_path), "--human-scores-csv", str(self.csv), "--output", str(output)]
        with patch.object(BLSPipeline, "_client", side_effect=AssertionError("must not call API")), redirect_stdout(io.StringIO()):
            self.assertEqual(benchmark_main(args), 0)
            first = output.read_bytes()
            with patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(benchmark_main(args), 2)
            self.assertEqual(output.read_bytes(), first)

    def test_pipeline_rejects_invalid_gold_before_inference(self):
        self.write_gold([self.gold_row("sample", 18)[:-1]])
        path = self.base / "sample.txt"
        path.write_text("合成発言", encoding="utf-8")
        pipeline = BLSPipeline(output_dir=self.base / "out", device_reference=None, human_scores_csv=self.csv)
        with patch.object(pipeline, "_client", side_effect=AssertionError("must not call API")):
            with self.assertRaises(ValueError):
                pipeline.run([path])

    def test_gold_changes_only_summary_and_reuses_cached_evaluation(self):
        self.write_gold([self.gold_row("sample", 18)])
        source = self.base / "sample.txt"
        source.write_text("合成発言", encoding="utf-8")
        payload = {"items": [{"id": i, "status": "not_detected", "reason": "合成テスト",
                              "evidence": [], "context": []} for i in range(1, 19)]}
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(status="completed", output_text=json.dumps(payload))
        client = SimpleNamespace(interactions=SimpleNamespace(create=create))
        pipeline = BLSPipeline(client=client, budget=SimpleNamespace(acquire=lambda *args: None),
                               output_dir=self.base / "out", device_reference=None, human_scores_csv=self.csv)
        first = pipeline.run([source])
        self.write_gold([self.gold_row("sample", 17)])
        second = pipeline.run([source])
        self.assertEqual(len(calls), 1)
        self.assertNotIn("模範", calls[0]["input"])
        self.assertEqual(first["results"][0]["result_path"], second["results"][0]["result_path"])
        self.assertIn("模範=100.0点 / 差=-100.0点", Path(first["summary_path"]).read_text())
        self.assertIn("模範=94.4点 / 差=-94.4点", Path(second["summary_path"]).read_text())

    def test_transcribe_only_skips_gold_and_summary(self):
        source = self.base / "sample.wav"
        source.write_bytes(b"synthetic")
        pipeline = BLSPipeline(output_dir=self.base / "out", device_reference=None, human_scores_csv=self.base / "missing.csv")
        with patch.object(pipeline, "transcribe", return_value={"key": "test", "input_sha256": "test"}):
            run = pipeline.run([source], transcription_only=True)
        self.assertEqual(run["status"], "completed")
        self.assertNotIn("summary_path", run)
