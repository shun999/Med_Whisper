from __future__ import annotations

import json
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.bls_scoring import evaluate_llm, evaluate_rules, load_rubric
from src.bls_validation import create_annotation_template, validate_predictions


COMPLETE_TRANSCRIPT = (
    "周囲の安全よし。傷病者発見、周囲の安全よし。感染防御よし。"
    "大丈夫ですか。大丈夫ですか。大丈夫ですか。誰か来てください。"
    "あなた119番に通報してください。あなたAEDを持ってきてください。"
    "呼吸の確認。脈の確認。胸骨圧迫を開始します。1、2、3、4、5、6。"
    "救急隊の方、この方は胸骨圧迫をしています。AEDで1回ショックをしました。"
    "この人の荷物は足元にあります。引き継ぎます。"
)


def calls_by_id(result):
    return {call.call_id: call for call in result.calls}


class RuleScoringTests(unittest.TestCase):
    def test_complete_transcript_scores_full_marks(self):
        result = evaluate_rules(COMPLETE_TRANSCRIPT)

        self.assertEqual(result.content_score, 100.0)
        self.assertEqual(result.sequence_score, 100.0)
        self.assertEqual(result.sequence_coverage, 6)
        self.assertTrue(all(call.score == 1.0 for call in result.calls))

    def test_partial_counts_and_undirected_requests(self):
        result = evaluate_rules(
            "大丈夫ですか。大丈夫ですか。119番通報してください。AEDを持ってきてください。"
        )
        calls = calls_by_id(result)

        self.assertEqual(calls["responsiveness_three_calls"].score, 0.5)
        self.assertEqual(calls["responsiveness_three_calls"].detected_count, 2)
        self.assertEqual(calls["request_emergency_call"].score, 0.5)
        self.assertEqual(calls["request_aed"].score, 0.5)

    def test_semantic_variants_are_detected(self):
        result = evaluate_rules(
            "周囲の安全を確認。感染対策します。聞こえますか、聞こえますか、聞こえますか。"
            "あなた、救急車を呼んでください。あなた、除細動器を取ってきてください。"
            "胸とお腹を見ます。頸動脈を確認します。"
        )
        calls = calls_by_id(result)

        self.assertEqual(calls["scene_safety"].score, 1.0)
        self.assertEqual(calls["infection_protection"].score, 1.0)
        self.assertEqual(calls["responsiveness_three_calls"].score, 1.0)
        self.assertEqual(calls["request_emergency_call"].score, 1.0)
        self.assertEqual(calls["request_aed"].score, 1.0)
        self.assertEqual(calls["check_breathing"].score, 1.0)
        self.assertEqual(calls["check_pulse"].score, 1.0)

    def test_negative_statement_is_not_scored(self):
        result = evaluate_rules("周囲の安全確認はしていません。感染防御していません。")
        calls = calls_by_id(result)

        self.assertEqual(calls["scene_safety"].score, 0.0)
        self.assertEqual(calls["infection_protection"].score, 0.0)

    def test_results_without_explicit_checks_receive_partial_credit(self):
        result = evaluate_rules("呼吸なし。脈ありません。")
        calls = calls_by_id(result)

        self.assertEqual(calls["check_breathing"].score, 0.5)
        self.assertEqual(calls["check_pulse"].score, 0.5)

    def test_short_compression_count_receives_partial_credit(self):
        result = evaluate_rules("胸骨圧迫を開始します。1、2、3。")
        call = calls_by_id(result)["compression_counting"]

        self.assertEqual(call.score, 0.5)
        self.assertEqual(call.detected_count, 3)

    def test_handoff_with_one_intervention_receives_partial_credit(self):
        result = evaluate_rules("救急隊の方、この方には胸骨圧迫をしています。引き継ぎます。")
        call = calls_by_id(result)["handoff_interventions"]

        self.assertEqual(call.score, 0.5)

    def test_reversed_order_fails_sequence_constraint(self):
        result = evaluate_rules("胸骨圧迫を開始します。1、2、3、4、5。呼吸の確認。脈の確認。")
        sequences = {item.constraint_id: item for item in result.sequences}

        self.assertEqual(sequences["vitals_before_compressions"].status, "fail")
        self.assertEqual(result.sequence_score, 0.0)

    def test_empty_transcript_is_safe_and_not_evaluable(self):
        result = evaluate_rules("")

        self.assertEqual(result.content_score, 0.0)
        self.assertIsNone(result.sequence_score)
        self.assertEqual(result.sequence_coverage, 0)
        self.assertTrue(all(call.score == 0.0 for call in result.calls))


class FakeInteractions:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(next(self.payloads), ensure_ascii=False))


class LlmScoringTests(unittest.TestCase):
    def test_three_runs_are_aggregated_and_disagreement_is_flagged(self):
        rubric = load_rubric()
        payloads = []
        for score in (0.0, 0.5, 0.5):
            payloads.append(
                {
                    "calls": [
                        {
                            "call_id": item["id"],
                            "score": score if item["id"] == "scene_safety" else 0.0,
                            "evidence": "周囲の安全よし" if item["id"] == "scene_safety" else "",
                            "reason": "test",
                            "confidence": 0.8,
                        }
                        for item in rubric["calls"]
                    ]
                }
            )
        interactions = FakeInteractions(payloads)
        client = SimpleNamespace(interactions=interactions)

        result = evaluate_llm("周囲の安全よし。", client, rubric=rubric, repeats=3)
        call = calls_by_id(result)["scene_safety"]

        self.assertEqual(call.score, 0.5)
        self.assertTrue(call.disagreement)
        self.assertAlmostEqual(call.score_stddev, 0.2357)
        self.assertAlmostEqual(call.run_agreement, 0.6667)
        self.assertEqual(call.char_start, 0)
        self.assertEqual(len(result.raw_runs), 3)
        self.assertEqual(len(interactions.requests), 3)
        self.assertEqual(
            interactions.requests[0]["response_format"]["mime_type"],
            "application/json",
        )

    def test_missing_llm_item_is_rejected(self):
        rubric = load_rubric()
        client = SimpleNamespace(interactions=FakeInteractions([{"calls": []}]))

        with self.assertRaisesRegex(ValueError, "評価項目が不足"):
            evaluate_llm("", client, rubric=rubric, repeats=1)


class ValidationTests(unittest.TestCase):
    def test_annotation_template_contains_text_and_audio_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["session_id", "sample_id", "transcript_path", "audio_path", "split"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "session_id": "session-1",
                        "sample_id": "sample-1",
                        "transcript_path": "transcript.txt",
                        "audio_path": "audio.wav",
                        "split": "test",
                    }
                )
            output = root / "annotations.csv"

            count = create_annotation_template(manifest, output, rubric_path="configs/bls_call_rubric.json")

            self.assertEqual(count, 72)
            self.assertTrue(output.is_file())

    def test_validation_reports_perfect_metrics_for_matching_labels(self):
        result = evaluate_rules(COMPLETE_TRANSCRIPT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "prediction.json"
            prediction.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-1",
                        "session_id": "session-1",
                        "source": "transcript.txt",
                        "results": [result.to_dict()],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            annotations = root / "annotations.csv"
            with annotations.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "session_id",
                        "sample_id",
                        "source_type",
                        "annotator_role",
                        "record_type",
                        "item_id",
                        "score",
                    ],
                )
                writer.writeheader()
                for call in result.calls:
                    writer.writerow(
                        {
                            "session_id": "session-1",
                            "sample_id": "sample-1",
                            "source_type": "text",
                            "annotator_role": "expert",
                            "record_type": "call",
                            "item_id": call.call_id,
                            "score": call.score,
                        }
                    )

            report = validate_predictions(
                annotations,
                [prediction],
                bootstrap_samples=10,
                seed=1,
            )

            metrics = report["comparisons"][0]["metrics"]
            self.assertEqual(metrics["call_exact_accuracy"], 1.0)
            self.assertEqual(metrics["call_macro_f1"], 1.0)
            self.assertEqual(metrics["content_score_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
