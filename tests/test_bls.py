"""Offline behavioral tests. Synthetic fixtures are not measured Gemini accuracy."""

import argparse
import copy
from contextlib import redirect_stdout
from datetime import datetime
import json
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.bls_evaluation import (InvalidResponse, build_prompt, clean_vocabulary, evaluate_transcript,
                                segments, validate_response)
from src.bls_pipeline import (BLSPipeline, MissingAPIKey, PACIFIC, QuotaExhausted, RequestBudget,
                             ROOT, create_gemini_client, digest, read_json, safe_output, write_json)
from src.bls_benchmark import character_errors, init_manifest, report, run_references
from src.bls_speech import separate_speech


def empty_payload():
    return {"items": [{"id": i, "status": "not_detected", "reason": "根拠未検出",
                       "evidence": [], "context": []} for i in range(1, 19)]}


def quote(text, value, *, kind="call", role="participant", occurrence=0):
    part = [s for s in segments(text) if value in s["text"]][occurrence]
    return {"segment_id": part["id"], "quote": value, "role": role, "kind": kind}


def mark(payload, i, evidence, context=None):
    payload["items"][i - 1].update(status="met", reason="発言に根拠あり", evidence=evidence, context=context or [])


class JudgeTests(unittest.TestCase):
    def test_equivalent_request_and_joint_findings(self):
        text = "救急車を呼んでください。脈なし、呼吸なし。"
        payload = empty_payload()
        mark(payload, 6, [quote(text, "救急車を呼んでください")])
        for i in (9, 10):
            mark(payload, i, [quote(text, "脈なし、呼吸なし")])
        result = evaluate_transcript(text, judge=lambda _: payload)
        self.assertEqual((result.met_count, result.score), (3, 16.7))

    def test_all_eighteen_can_pass_without_resumption_declaration(self):
        utterances = ["傷病者発見", "周囲の安全よし", "感染防御よし", "大丈夫ですか", "誰か来てください",
                      "119番に通報してください", "AEDを持ってきてください", "必ず戻ってきてください",
                      "呼吸なし", "脈なし", "胸骨圧迫、1、2、3、4", "AEDを使えますか", "胸骨圧迫を代わってください",
                      "1、2、3で交代", "皆さん下がってください", "5、6、7、8",
                      "救急隊の方、ここで倒れているところを発見しました", "こちらが傷病者の荷物です"]
        text = "。".join(utterances[:15] + ["放電が終了しました"] + utterances[15:]) + "。"
        payload = empty_payload()
        shock = quote(text, "放電が終了しました", kind="shock_complete", role="device")
        for i, utterance in enumerate(utterances, 1):
            kind = "count" if i in (11, 16) else "handoff" if i == 14 else "call"
            mark(payload, i, [quote(text, utterance, kind=kind)], [shock] if i in (15, 16) else [])
        self.assertEqual(validate_response(text, payload).score, 100)

    def test_hallucinated_quote_becomes_uncertain(self):
        text = "傷病者発見。"
        payload = empty_payload()
        mark(payload, 3, [{"segment_id": 1, "quote": "感染防御よし", "role": "participant", "kind": "call"}])
        result = validate_response(text, payload)
        self.assertEqual(result.items[2].status, "uncertain")
        self.assertEqual(result.score, 0)

    def test_invalid_shape_is_failure_not_zero_score(self):
        for payload in ({}, {"items": []}, {"items": empty_payload()["items"][:-1]}):
            with self.assertRaises(InvalidResponse):
                validate_response("発言", payload)
        payload = empty_payload()
        payload["items"][1]["id"] = 1
        with self.assertRaises(InvalidResponse):
            validate_response("発言", payload)
        payload = empty_payload()
        payload["items"][0]["id"] = True
        with self.assertRaises(InvalidResponse):
            validate_response("発言", payload)

    def test_empty_input_is_failure(self):
        with self.assertRaises(ValueError):
            evaluate_transcript(" \n", judge=lambda _: empty_payload())

    def test_device_and_instructor_never_pass(self):
        text = "離れてください。放電が終了しました。"
        for role in ("device", "instructor", "unknown"):
            payload = empty_payload()
            mark(payload, 15, [quote(text, "離れてください", role=role)],
                 [quote(text, "放電が終了しました", kind="shock_complete", role="device")])
            self.assertEqual(validate_response(text, payload).items[14].status, "uncertain")

    def test_ambiguous_device_sequence_even_if_model_claims_participant(self):
        text = "充電中です。体から離れてください。ショックを実行します。ショックが完了しました。"
        payload = empty_payload()
        mark(payload, 15, [quote(text, "体から離れてください")],
             [quote(text, "ショックを実行します", kind="shock", role="device")])
        self.assertEqual(validate_response(text, payload).items[14].status, "uncertain")

    def test_use_intention_is_not_ability(self):
        text = "あなたはAEDを使いますか？"
        payload = empty_payload()
        mark(payload, 12, [quote(text, "AEDを使いますか")])
        self.assertEqual(validate_response(text, payload).items[11].status, "uncertain")

    def test_negated_request_not_awarded(self):
        text = "救急車を呼ばないでください。"
        payload = empty_payload()
        mark(payload, 6, [quote(text, "救急車を呼ばないでください")])
        self.assertEqual(validate_response(text, payload).score, 0)

    def test_count_after_shock_requires_order_and_completion(self):
        text = "1、2、3、4。放電終了。5、6、7、8。"
        for value, expected in (("1、2、3、4", "uncertain"), ("5、6、7、8", "met")):
            payload = empty_payload()
            mark(payload, 16, [quote(text, value, kind="count")],
                 [quote(text, "放電終了", kind="shock_complete", role="device")])
            self.assertEqual(validate_response(text, payload).items[15].status, expected)
        payload["items"][15]["context"] = []
        self.assertEqual(validate_response(text, payload).items[15].status, "uncertain")

    def test_shock_prediction_cannot_be_mislabeled_completion(self):
        text = "電気ショックが必要です。1、2、3、4。"
        payload = empty_payload()
        mark(payload, 16, [quote(text, "1、2、3、4", kind="count")],
             [quote(text, "電気ショックが必要です", kind="shock_complete", role="device")])
        self.assertEqual(validate_response(text, payload).items[15].status, "uncertain")

    def test_handoff_count_reuse_is_not_double_credit(self):
        text = "1、2、3で交代。"
        payload = empty_payload()
        mark(payload, 11, [quote(text, "1、2、3", kind="count")])
        mark(payload, 14, [quote(text, "1、2、3で交代", kind="handoff")])
        result = validate_response(text, payload)
        self.assertEqual((result.items[10].status, result.items[13].status), ("uncertain", "uncertain"))

    def test_compression_declaration_without_count_does_not_pass(self):
        text = "胸骨圧迫を始めます。"
        payload = empty_payload()
        mark(payload, 11, [quote(text, "胸骨圧迫を始めます", kind="count")])
        self.assertEqual(validate_response(text, payload).items[10].status, "uncertain")

    def test_raw_offsets_and_normalized_candidates(self):
        text = " \nＡＥＤ\u200bを持ってきてください！\n脈無し、呼吸無し。"
        prompt = json.loads(build_prompt(text))
        self.assertEqual(prompt["rule_candidates"]["7"], [1])
        for segment in prompt["segments"]:
            self.assertEqual(text[segment["start"]:segment["end"]], segment["text"])
        self.assertEqual(clean_vocabulary(["ＡＥＤ\u200b", "AED", " "]), ["AED"])


class SpeechSeparationTests(unittest.TestCase):
    reference = ("パッドのコネクタを接続してください。充電中です。体から離れてください。"
                 "ショックを実行中です。ショックが完了しました。"
                 "ただちに胸骨圧迫と人工呼吸をしてください。")

    def speech_quote(self, text, value, *, kind="call", source_role=None):
        parts = separate_speech(text, self.reference)["segments"]
        part = next(p for p in parts if value in p["text"] and
                    (source_role is None or p["source_role"] == source_role))
        return {"segment_id": part["id"], "quote": value, "role": "participant", "kind": kind}

    def test_mixed_sentence_keeps_count_and_distinct_human_call(self):
        text = "123充電中です。体から離れてください。離れてください。ショックが完了しました。4、5、6。"
        speech = separate_speech(text, self.reference)
        self.assertEqual(speech["human_candidate_text"], "123\n離れてください。\n4、5、6。")
        for part in speech["segments"]:
            self.assertEqual(text[part["start"]:part["end"]], part["text"])
        payload = empty_payload()
        completion = self.speech_quote(text, "ショックが完了しました", kind="shock_complete")
        mark(payload, 15, [self.speech_quote(text, "離れてください", source_role="candidate")], [completion])
        mark(payload, 16, [self.speech_quote(text, "4、5、6", kind="count")], [completion])
        result = validate_response(text, payload, device_reference=self.reference)
        self.assertEqual([result.items[i].status for i in (14, 15)], ["met", "met"])
        self.assertEqual(result.items[15].context[0]["role"], "device")

    def test_device_quote_cannot_be_reclassified_as_participant(self):
        text = self.reference
        payload = empty_payload()
        mark(payload, 15, [self.speech_quote(text, "離れてください")],
             [self.speech_quote(text, "ショックが完了しました", kind="shock_complete")])
        mark(payload, 9, [self.speech_quote(text, "人工呼吸")])
        result = validate_response(text, payload, device_reference=self.reference)
        self.assertEqual(result.score, 0)
        self.assertEqual(result.items[14].evidence[0]["role"], "device")
        self.assertEqual(result.items[14].status, "uncertain")

    def test_device_prompt_has_no_scoring_candidates_or_unheard_reference(self):
        text = "充電中です。体から離れてください。"
        prompt = json.loads(build_prompt(text, device_reference=self.reference))
        self.assertEqual(prompt["transcript"], "")
        self.assertEqual(prompt["segments"], [])
        self.assertTrue(all(not candidates for candidates in prompt["rule_candidates"].values()))
        self.assertEqual(len(prompt["device_context"]), 2)
        self.assertNotIn("ショックが完了しました", json.dumps(prompt, ensure_ascii=False))

    def test_spelling_normalization_preserves_original_offsets(self):
        text = "ﾊﾟｯﾄﾞのコネクターを、接続してください。\n直ちに胸骨圧迫と人工呼吸をして\u200bください。"
        speech = separate_speech(text, self.reference)
        self.assertEqual(speech["human_candidate_text"], "")
        self.assertIn("コネクター", speech["device_text"])
        for part in speech["segments"]:
            self.assertEqual(text[part["start"]:part["end"]], part["text"])

    def test_reference_matching_prefers_longest_phrase_and_all_occurrences(self):
        speech = separate_speech("パッドのコネクタを接続してください。コネクタを接続してください。",
                                 "コネクタを接続してください。パッドのコネクタを接続してください。")
        self.assertEqual(speech["human_candidate_text"], "")
        self.assertEqual(len(speech["segments"]), 2)

    def test_explicit_human_label_preserves_identical_device_phrase(self):
        text = "AED: 充電中です。体から離れてください。\n参加者1: 体から離れてください。\nAED: ショックが完了しました。"
        speech = separate_speech(text, self.reference)
        self.assertEqual(speech["human_candidate_text"], "参加者1: 体から離れてください。")
        payload = empty_payload()
        mark(payload, 15, [self.speech_quote(text, "体から離れてください", source_role="participant")],
             [self.speech_quote(text, "ショックが完了しました", kind="shock_complete")])
        self.assertEqual(validate_response(text, payload, device_reference=self.reference).items[14].status, "met")

    def test_instructor_unknown_and_label_scope(self):
        text = "指導者: 救急車を呼んでください。\n不明: AEDを持ってきてください。\n誰か来てください。"
        speech = separate_speech(text, self.reference)
        self.assertEqual(speech["human_candidate_text"], "誰か来てください。")
        payload = empty_payload()
        mark(payload, 6, [self.speech_quote(text, "救急車を呼んでください")])
        mark(payload, 7, [self.speech_quote(text, "AEDを持ってきてください")])
        self.assertEqual(validate_response(text, payload, device_reference=self.reference).score, 0)

    def test_unresolved_device_sequence_stays_uncertain(self):
        text = "充電中です。離れてください。ショックを実行します。ショックが完了しました。"
        payload = empty_payload()
        mark(payload, 15, [self.speech_quote(text, "離れてください")],
             [self.speech_quote(text, "ショックが完了しました", kind="shock_complete")])
        self.assertEqual(validate_response(text, payload, device_reference=self.reference).items[14].status, "uncertain")

    def test_execution_is_not_shock_completion(self):
        text = "ショックを実行中です。1、2、3。"
        payload = empty_payload()
        mark(payload, 16, [self.speech_quote(text, "1、2、3", kind="count")],
             [self.speech_quote(text, "ショックを実行中です", kind="shock_complete")])
        self.assertEqual(validate_response(text, payload, device_reference=self.reference).items[15].status, "uncertain")


class FakeBudget:
    def __init__(self):
        self.requests = []

    def acquire(self, *args):
        self.requests.append(args)


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.uploads = []
        self.deletes = []
        self.interactions = SimpleNamespace(create=self.create)
        self.files = SimpleNamespace(upload=self.upload, delete=lambda **kw: self.deletes.append(kw))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(output_text=response, status="completed")

    def upload(self, **kwargs):
        self.uploads.append(kwargs)
        self.assert_ascii = str(kwargs["file"].name).isascii()
        return SimpleNamespace(name="files/synthetic", uri="test://audio", mime_type=kwargs["config"]["mime_type"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.device_reference = self.base / "device.txt"
        self.device_reference.write_text("装置を準備しています。", encoding="utf-8")

    def pipeline(self, client, **kwargs):
        kwargs.setdefault("device_reference", self.device_reference)
        return BLSPipeline(client=client, budget=FakeBudget(), output_dir=self.base / "output", **kwargs)

    def test_default_request_limits(self):
        pipeline = self.pipeline(FakeClient([]))
        self.assertEqual(pipeline.limits["transcription"], (2, 100))
        self.assertEqual(pipeline.limits["evaluation"], (1000, 10_000))

        from scripts.evaluate_bls import pipeline_arguments
        parser = argparse.ArgumentParser()
        pipeline_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual((args.transcription_rpm, args.transcription_rpd), (2, 100))
        self.assertEqual((args.evaluation_rpm, args.evaluation_rpd), (1000, 10_000))

    def test_text_cache_and_exports_need_no_second_request(self):
        source = self.base / "example.txt"
        source.write_text("傷病者発見。", encoding="utf-8")
        client = FakeClient([json.dumps(empty_payload())])
        pipeline = self.pipeline(client)
        a, b = pipeline.evaluate_file(source), pipeline.evaluate_file(source)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(a["result_path"], b["result_path"])
        self.assertTrue(Path(a["result_path"]).with_suffix(".csv").exists())
        self.assertEqual(client.calls[0]["response_format"]["mime_type"], "application/json")

    def test_models_change_cache_keys(self):
        client = FakeClient([json.dumps(empty_payload()), json.dumps(empty_payload())])
        a = self.pipeline(client).evaluate_text("発言")
        b = self.pipeline(client, evaluation_model="another-model").evaluate_text("発言")
        self.assertNotEqual(a["cache_key"], b["cache_key"])

    def test_device_reference_invalidates_only_evaluation_and_exports_separation(self):
        source = self.base / "synthetic.wav"
        source.write_bytes(b"synthetic audio")
        text = "装置を準備しています。誰か来てください。"
        client = FakeClient([text, json.dumps(empty_payload()), json.dumps(empty_payload())])
        pipeline = self.pipeline(client)
        first = pipeline.evaluate_file(source)
        cached = pipeline.evaluate_file(source)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(first["result_path"], cached["result_path"])
        self.assertEqual(Path(first["result_path"]).with_suffix(".human.txt").read_text(), "誰か来てください。")
        self.assertEqual(Path(first["result_path"]).with_suffix(".device.txt").read_text(), "装置を準備しています。")
        self.device_reference.write_text("別の装置のアナウンスです。", encoding="utf-8")
        second = pipeline.evaluate_file(source)
        self.assertEqual(len(client.uploads), 1)
        self.assertEqual(len(client.calls), 3)
        self.assertNotEqual(first["cache_key"], second["cache_key"])
        self.assertEqual(first["transcription"]["key"], second["transcription"]["key"])
        # Benchmark uses the saved reference even after the live file changes.
        manifest = {"samples": [{"sample_id": source.stem, "audio_sha256": first["input_sha256"],
                                "split": "validation", "labels": {str(i): False for i in range(1, 19)}}]}
        run = {"status": "completed", "results": [{k: first[k] for k in ("sample_id", "input_sha256", "result_path")}]}
        self.assertEqual(report(manifest, run)["score_mae"], 0)

    def test_missing_or_empty_device_reference_fails_before_api(self):
        source = self.base / "synthetic.wav"
        source.write_bytes(b"synthetic audio")
        client = FakeClient([])
        for reference in (self.base / "missing.txt", self.device_reference):
            self.device_reference.write_text("。\n", encoding="utf-8")
            pipeline = self.pipeline(client, device_reference=reference)
            with self.assertRaises(ValueError):
                pipeline.evaluate_file(source)
            self.assertEqual(pipeline.budget.requests, [])
        self.assertEqual(client.calls, [])
        self.assertEqual(client.uploads, [])

    def test_cli_can_run_from_cache_without_credentials(self):
        from scripts.evaluate_bls import main
        source = self.base / "cached.txt"
        source.write_text("傷病者発見。", encoding="utf-8")
        self.pipeline(FakeClient([json.dumps(empty_payload())])).evaluate_file(source)
        output = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), redirect_stdout(output):
            code = main(["--transcript", str(source), "--output-dir", str(self.base / "output"),
                         "--device-reference", str(self.device_reference)])
        self.assertEqual(code, 0)
        self.assertIn("completed", output.getvalue())

    def test_equal_contents_reuse_inference_but_preserve_each_source_path(self):
        client = FakeClient([json.dumps(empty_payload())])
        pipeline = self.pipeline(client)
        records = []
        for directory in ("first", "second"):
            path = self.base / directory / "sample.txt"
            path.parent.mkdir()
            path.write_text("傷病者発見。", encoding="utf-8")
            records.append(pipeline.evaluate_file(path))
        self.assertEqual(len(client.calls), 1)
        self.assertNotEqual(records[0]["source_path"], records[1]["source_path"])
        self.assertNotEqual(records[0]["result_path"], records[1]["result_path"])

    def test_audio_cleanup_cache_and_vocabulary_invalidation(self):
        source = self.base / "日本語.wav"
        source.write_bytes(b"synthetic audio for mocked upload")
        original = source.read_bytes()
        client = FakeClient(["傷病者発見。", "傷病者発見。"])
        a = self.pipeline(client)
        first = a.transcribe(source)
        self.assertEqual(a.transcribe(source), first)
        self.pipeline(client, vocabulary="baseline").transcribe(source)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(client.deletes), 2)
        self.assertTrue(client.assert_ascii)
        self.assertEqual(source.read_bytes(), original)
        self.assertNotEqual(client.calls[0]["generation_config"], client.calls[1]["generation_config"])

    def test_audio_cleanup_on_request_error(self):
        source = self.base / "日本語.wav"
        source.write_bytes(b"synthetic")
        client = FakeClient([RuntimeError("synthetic failure")])
        with self.assertRaises(RuntimeError):
            self.pipeline(client).transcribe(source)
        self.assertEqual(len(client.deletes), 1)

    def test_bad_json_archived_but_not_reused_as_successful_cache(self):
        client = FakeClient(["not JSON", json.dumps(empty_payload())])
        pipeline = self.pipeline(client)
        with self.assertRaises(InvalidResponse):
            pipeline.evaluate_text("発言")
        pipeline.evaluate_text("発言")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(list((self.base / "output" / "responses").glob("*.json"))), 2)

    def test_missing_key_does_not_consume_budget(self):
        pipeline = self.pipeline(None)
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(MissingAPIKey):
            pipeline.evaluate_text("発言")
        self.assertEqual(pipeline.budget.requests, [])

    def test_failed_request_has_no_score_and_resume_reuses_success(self):
        a, b = self.base / "a.txt", self.base / "b.txt"
        a.write_text("発言A", encoding="utf-8")
        b.write_text("発言B", encoding="utf-8")
        client = FakeClient([json.dumps(empty_payload()), QuotaExhausted("day limit"), json.dumps(empty_payload())])
        pipeline = self.pipeline(client)
        first = pipeline.run([a, b])
        self.assertEqual(first["status"], "interrupted")
        self.assertEqual(len(first["results"]), 1)
        self.assertEqual(first["pending"], [str(b)])
        second = pipeline.run([a, b])
        self.assertEqual(second["status"], "completed")
        self.assertEqual(len(client.calls), 3)

    def test_retry_429_is_bounded_and_counted(self):
        error = RuntimeError("rate limited")
        error.code = 429
        client = FakeClient([error, error, json.dumps(empty_payload())])
        delays = []
        pipeline = self.pipeline(client, sleep=delays.append)
        pipeline.evaluate_text("発言")
        self.assertEqual(len(pipeline.budget.requests), 3)
        self.assertEqual(delays, [5, 10])

    def test_output_protection_including_symlink(self):
        for path in (ROOT / "data" / "new.json", ROOT / "env" / "new.json", ROOT / "gemini.ipynb"):
            with self.assertRaises(ValueError):
                safe_output(path)
        link = self.base / "linked"
        link.symlink_to(ROOT / "data", target_is_directory=True)
        with self.assertRaises(ValueError):
            write_json(link / "new.json", {})

    def test_notebook_cells_execute_with_synthetic_cached_pipeline(self):
        import nbformat
        notebook = nbformat.read(ROOT / "gemini.ipynb", as_version=4)
        nbformat.validate(notebook)
        source = self.base / "synthetic.txt"
        source.write_text("傷病者発見。", encoding="utf-8")
        pipeline = self.pipeline(FakeClient([json.dumps(empty_payload())]))
        namespace = {}
        with (patch.dict("os.environ", {}, clear=True), patch("getpass.getpass", return_value=""),
              patch("src.bls_pipeline.BLSPipeline", return_value=pipeline),
              patch("IPython.display.display"), redirect_stdout(io.StringIO())):
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    exec(compile(cell.source, f"gemini.ipynb:{cell.id}", "exec"), namespace)
                    if cell.id == "paths":
                        namespace["INPUT_PATH"] = source
        self.assertEqual(namespace["evaluation"]["met_count"], 0)
        self.assertEqual(len(pipeline.client.calls), 1)


class BudgetTests(unittest.TestCase):
    def test_durable_minute_and_pacific_day_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            timestamp = [datetime(2026, 9, 7, 12, tzinfo=PACIFIC).timestamp()]
            delays = []
            def sleep(delay):
                delays.append(delay)
                timestamp[0] += delay
            def budget():
                return RequestBudget(Path(folder) / "budget", clock=lambda: timestamp[0], sleep=sleep)
            budget().acquire("model", 1, 2)
            budget().acquire("model", 1, 2)
            self.assertGreaterEqual(sum(delays), 60)
            with self.assertRaises(QuotaExhausted):
                budget().acquire("model", 1, 2)
            budget().acquire("separate-model", 1, 2)
            timestamp[0] = datetime(2026, 9, 8, 0, 1, tzinfo=PACIFIC).timestamp()
            budget().acquire("model", 1, 2)


class SDKContractTests(unittest.TestCase):
    def test_transcription_config_survives_sdk_serialization(self):
        import httpx
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "completed", "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "傷病者発見。"}]}]})
        with tempfile.TemporaryDirectory() as folder, httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = create_gemini_client("offline-test-key", httpx_client=http)
            self.addCleanup(client.close)
            pipeline = BLSPipeline(client=client, budget=FakeBudget(), output_dir=Path(folder) / "out")
            config = {"language_codes": ["ja-JP"], "custom_vocabulary": ["胸骨圧迫"], "mode": {"type": "verbatim"}}
            response = pipeline._request("transcription", model=pipeline.transcription_model,
                input=[{"type": "audio", "uri": "test://audio", "mime_type": "audio/wav"}],
                generation_config={"transcription_config": config})
            self.assertEqual(response.output_text, "傷病者発見。")
            self.assertEqual(requests[0]["generation_config"]["transcription_config"], config)

    def test_installed_sdk_structured_output_and_no_hidden_retries(self):
        import httpx
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            if len(requests) < 3:
                return httpx.Response(429, json={"error": {"code": 429, "message": "synthetic rate limit", "status": "RESOURCE_EXHAUSTED"}})
            return httpx.Response(200, json={"id": "offline", "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": json.dumps(empty_payload())}]}]})
        with tempfile.TemporaryDirectory() as folder, httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = create_gemini_client("offline-test-key", httpx_client=http)
            self.addCleanup(client.close)
            budget = FakeBudget()
            pipeline = BLSPipeline(client=client, budget=budget, output_dir=Path(folder) / "out", sleep=lambda _: None,
                                   device_reference=None)
            result = pipeline.evaluate_text("発言")
            self.assertEqual(result["evaluation"]["met_count"], 0)
            self.assertEqual(len(requests), 3)
            self.assertEqual(len(budget.requests), 3)
            self.assertEqual(requests[-1]["response_format"]["mime_type"], "application/json")
            self.assertIn("schema", requests[-1]["response_format"])
            items_schema = requests[-1]["response_format"]["schema"]["properties"]["items"]
            self.assertNotIn("minItems", items_schema)
            self.assertNotIn("maxItems", items_schema)
            self.assertIn("system_instruction", requests[-1])


class BenchmarkTests(unittest.TestCase):
    def test_character_error_rate_ignores_width_and_whitespace_only(self):
        self.assertEqual(character_errors("ＡＥＤ\nです", "AEDです"), (0, 5))
        self.assertEqual(character_errors("脈なし", "脈あり"), (2, 3))

    def test_unknown_gold_never_inferred_from_filename_elsewhere(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "20260706_2回目.wav").write_bytes(b"synthetic")
            manifest = init_manifest(path, path / "gold.json")
            self.assertTrue(all(v is None for v in manifest["samples"][0]["labels"].values()))
            with self.assertRaises(FileExistsError):
                init_manifest(path, path / "gold.json")

    def test_report_coverage_errors_and_reference_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            client = FakeClient([json.dumps(empty_payload()), json.dumps(empty_payload())])
            pipeline = BLSPipeline(client=client, budget=FakeBudget(), output_dir=path / "output", device_reference=None)
            text = "救急車を呼んでください。"
            document = pipeline.evaluate_text(text)
            record = pipeline.export(document, sample_id="sample", source_path="synthetic.wav", input_sha256="audiohash", source_kind="audio")
            run = {"status": "completed", "results": [{k: record[k] for k in ("sample_id", "input_sha256", "result_path")}]}
            sample = {"sample_id": "sample", "audio_sha256": "audiohash", "split": "validation",
                      "labels": {str(i): i == 6 for i in range(1, 19)}, "reference_text": text}
            manifest = {"samples": [sample, {**copy.deepcopy(sample), "sample_id": "missing"}]}
            result = report(manifest, run)
            self.assertEqual(result["evaluated_labeled_items"], 18)
            self.assertEqual(result["micro"]["fn"], 1)
            self.assertEqual(result["micro"]["tn"], 17)
            self.assertEqual(result["score_mae"], 5.6)
            self.assertEqual(result["cer"], 0)
            self.assertFalse(result["coverage"][1]["evaluated"])
            ref_run = run_references({"samples": [sample]}, pipeline, path / "references.json")
            comparison = report(manifest, run, reference_run=ref_run)
            self.assertEqual(comparison["reference_evaluator_errors"], 1)
            self.assertEqual(comparison["errors"][0]["stage_hint"], "evaluator_also_failed")
            sample["audio_sha256"] = "different"
            with self.assertRaises(ValueError):
                report(manifest, run)

    def test_no_gold_produces_no_metrics(self):
        with self.assertRaises(ValueError):
            report({"samples": [{"sample_id": "none", "audio_sha256": "none", "split": "validation",
                                  "labels": {str(i): None for i in range(1, 19)}}]},
                   {"status": "completed", "results": []})


if __name__ == "__main__":
    unittest.main()
