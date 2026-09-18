"""Regression coverage for controlled scoring and transcription experiments."""

import copy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.compare_bls import compare_metrics
from src.bls_benchmark import import_human_scores, report
from src.bls_evaluation import build_prompt
from src.bls_pipeline import BLSPipeline
from src.bls_speech import extract_word_annotations
from test_bls import FakeBudget, FakeClient, empty_payload


class AnnotationTests(unittest.TestCase):
    text = "離れてください。離れてください。"

    def raw(self):
        return {"steps": [{"content": [{"annotations": [
            {"type": "word_info", "text": "離れてください", "speaker": "spk_1", "start_offset": "1.2s", "end_offset": "2s"},
            {"type": "word_info", "text": "離れてください", "speaker": "spk_2", "start_offset": "3s", "end_offset": "4s"},
        ]}]}]}

    def test_repeated_words_keep_separate_offsets_and_acoustic_times(self):
        words = extract_word_annotations(self.raw(), self.text)
        self.assertEqual([w["start"] for w in words], [0, 8])
        self.assertEqual(words[0]["start_seconds"], 1.2)
        self.assertEqual(words[1]["speaker"], "spk_2")
        for word in words:
            self.assertEqual(self.text[word["start"]:word["end"]], word["text"])
        prompt = json.loads(build_prompt(self.text, word_annotations=words))
        self.assertEqual(prompt["word_annotations"], words)
        self.assertIn("参加者役の確定ラベルではない", prompt["annotation_guidance"])

    def test_bad_annotations_do_not_become_guessed_alignment(self):
        with self.assertRaises(ValueError):
            extract_word_annotations({}, self.text)
        for field, value in (("text", "存在しない"), ("start_offset", "nan"),
                             ("end_offset", "0s"), ("speaker", 42)):
            raw = self.raw()
            raw["steps"][0]["content"][0]["annotations"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                extract_word_annotations(raw, self.text)

    def test_speaker_request_omits_vocabulary_and_has_separate_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            audio = base / "clip.wav"
            audio.write_bytes(b"synthetic")
            client = FakeClient([])
            response = SimpleNamespace(output_text=self.text, status="completed", model_dump=lambda **_: self.raw())
            common = dict(output_dir=base / "outputs", client=client, budget=FakeBudget(), device_reference=None,
                          human_scores_csv=None)
            normal = BLSPipeline(**common)
            speakers = BLSPipeline(**common, transcription_profile="speakers", vocabulary="none")
            with patch.object(client.interactions, "create", return_value=response) as request:
                standard = normal.transcribe(audio)
                annotated = speakers.transcribe(audio)
                cached = speakers.transcribe(audio)
            self.assertEqual(request.call_count, 2)
            config = request.call_args.kwargs["generation_config"]["transcription_config"]
            self.assertNotIn("custom_vocabulary", config)
            self.assertEqual(config["mode"]["timestamp_granularities"], ["word"])
            self.assertEqual(annotated, cached)
            self.assertNotEqual(standard["key"], annotated["key"])
            self.assertEqual(len(annotated["word_annotations"]), 2)

    def test_conflicting_features_fail_before_api_calls(self):
        with self.assertRaisesRegex(ValueError, "併用"):
            BLSPipeline(transcription_profile="speakers", vocabulary="targeted")

    def test_missing_annotations_preserve_raw_response_but_not_success_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            audio = base / "clip.wav"
            audio.write_bytes(b"synthetic")
            client = FakeClient([self.text])
            pipeline = BLSPipeline(output_dir=base / "outputs", client=client, budget=FakeBudget(),
                                   transcription_profile="speakers", vocabulary="none")
            with self.assertRaisesRegex(ValueError, "単語注釈"):
                pipeline.transcribe(audio)
            self.assertEqual(len(list((base / "outputs/responses").glob("transcription_*.json"))), 1)
            self.assertEqual(list((base / "outputs/cache/transcription").glob("*.json")), [])
            self.assertEqual(len(client.deletes), 1)

    def test_audio_annotations_invalidate_evaluation_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient([json.dumps(empty_payload())] * 2)
            pipeline = BLSPipeline(output_dir=Path(temp) / "outputs", client=client, budget=FakeBudget(),
                                   device_reference=None, human_scores_csv=None)
            plain = pipeline.evaluate_text(self.text)
            annotated = pipeline.evaluate_text(self.text, word_annotations=extract_word_annotations(self.raw(), self.text))
            self.assertNotEqual(plain["cache_key"], annotated["cache_key"])
            self.assertEqual(len(client.calls), 2)


class LabelImportTests(unittest.TestCase):
    def test_import_is_explicit_and_does_not_fabricate_transcripts(self):
        manifest = {"samples": [{"sample_id": "sample", "split": "development",
                                 "labels": {str(i): None for i in range(1, 19)}, "reference_text": None}]}
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "scores.csv"
            rows = io.StringIO()
            writer = csv.writer(rows)
            writer.writerow(["BLS評価対象コール", *[str(i) for i in range(18)], "成功項目数", "点数"])
            writer.writerow(["sample", *([1] * 17), 0, 17, 94.4])
            source.write_text(rows.getvalue())
            result = import_human_scores(manifest, source)
            self.assertIsNone(manifest["samples"][0]["labels"]["1"])
            self.assertIsNone(result["samples"][0]["reference_text"])
            self.assertFalse(result["samples"][0]["labels"]["18"])
            self.assertTrue(result["human_scores_csv"]["sha256"])
            conflict = copy.deepcopy(manifest)
            conflict["samples"][0]["labels"]["18"] = True
            with self.assertRaisesRegex(ValueError, "矛盾"):
                import_human_scores(conflict, source)


class ExperimentTests(unittest.TestCase):
    def test_historical_reports_are_explicit_and_check_score_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            pipeline = BLSPipeline(output_dir=Path(temp) / "outputs", client=FakeClient([json.dumps(empty_payload())]),
                                   budget=FakeBudget(), device_reference=None, human_scores_csv=None)
            record = pipeline.export(pipeline.evaluate_text("合成テキスト。"), sample_id="sample", source_path="synthetic.wav",
                                     input_sha256="synthetic-audio-hash", source_kind="audio")
            path = Path(record.pop("result_path"))
            record["evaluation"]["validator_version"] = "historical-validator"
            path.write_text(json.dumps(record))
            run = {"status": "completed", "results": [{"sample_id": "sample", "input_sha256": "synthetic-audio-hash",
                                                         "result_path": str(path)}]}
            manifest = {"samples": [{"sample_id": "sample", "audio_sha256": "synthetic-audio-hash", "split": "validation",
                                     "labels": {str(i): False for i in range(1, 19)}}]}
            measured = report(manifest, run, saved_results=True)
            self.assertEqual(measured["verification"], "saved_snapshot")
            self.assertEqual(measured["score_mae"], 0)
            with self.assertRaisesRegex(ValueError, "現在の検証器"):
                report(manifest, run)
            record["evaluation"]["score"] = 100
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "不正"):
                report(manifest, run, saved_results=True)

    def measurement(self, mae, fp=0):
        return {"scores": [{"sample_id": "a"}], "run_status": "completed", "score_mae": mae,
                "evaluated_labeled_items": 18, "micro": {"fp": fp}}

    def test_lower_mae_with_false_positives_is_not_accepted(self):
        baseline = self.measurement(4.18)
        self.assertTrue(compare_metrics(baseline, self.measurement(2))["improves_baseline"])
        self.assertFalse(compare_metrics(baseline, self.measurement(2, 1))["improves_baseline"])
        self.assertFalse(compare_metrics(baseline, self.measurement(4.18))["improves_baseline"])

    def test_partial_or_different_sample_coverage_is_not_comparable(self):
        baseline = self.measurement(4.18)
        candidate = self.measurement(0)
        candidate["scores"] = []
        self.assertFalse(compare_metrics(baseline, candidate)["comparable"])
        candidate = self.measurement(0)
        candidate["run_status"] = "interrupted"
        self.assertFalse(compare_metrics(baseline, candidate)["improves_baseline"])
