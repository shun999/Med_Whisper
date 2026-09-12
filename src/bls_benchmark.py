"""Gold-label templates and explicit-coverage benchmark reports; never infer gold."""

import re
from pathlib import Path

from src.bls_evaluation import CRITERIA, normalize, validate_response
from src.bls_pipeline import ROOT, digest, file_digest, now_iso, read_json, write_json
from src.bls_speech import separate_speech


def init_manifest(audio_dir: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError("既存の正解ラベルは上書きしません")
    files = sorted(Path(audio_dir).glob("*.wav"))
    if not files:
        raise ValueError("WAVがありません")
    samples = []
    for path in files:
        match = re.fullmatch(r"20260706_(\d+)回目", path.stem)
        official = path.parent.resolve() == (ROOT / "data" / "20260706").resolve() and match is not None
        number = int(match[1]) if official else None
        known = number in (2, 3, 4, 5)
        split = "validation" if number in (4, 5, *range(13, 21)) else "development"
        samples.append({"sample_id": path.stem, "audio_path": str(path.resolve()),
                        "audio_sha256": file_digest(path), "split": split,
                        "labels": {str(i): True if known else None for i in range(1, 19)},
                        "label_source": "AGENTS.md" if known else "pending",
                        "reference_text": None,
                        "reference_evidence": {str(i): None for i in range(1, 19)}})
    manifest = {"version": 1, "created_at": now_iso(), "samples": samples,
                "note": "nullは未注釈。falseは音声確認で未実施と判断した場合だけ記入。reference_textは人が音声確認した逐語録。"}
    write_json(destination, manifest)
    return manifest


def validate_manifest(manifest: dict) -> list[dict]:
    samples = manifest.get("samples", [])
    if not samples or len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("正解データが空、またはsample_idが重複しています")
    for sample in samples:
        labels = sample["labels"]
        if set(labels) != {str(i) for i in range(1, 19)} or any(v is not None and type(v) is not bool for v in labels.values()):
            raise ValueError("正解は18項目それぞれtrue/false/nullで指定してください")
        if sample["split"] not in ("development", "validation"):
            raise ValueError("splitが不正です")
        if sample.get("reference_text") is not None and not isinstance(sample["reference_text"], str):
            raise ValueError("reference_textは文字列またはnullです")
    return samples


def _load_run(run: dict) -> dict:
    if run.get("stage") == "transcription":
        raise ValueError("文字起こし専用runは採点精度の比較に使えません")
    results = {}
    for entry in run["results"]:
        sample_id = entry["sample_id"]
        if sample_id in results:
            raise ValueError(f"runに同じsample_idが複数あります: {sample_id}")
        record = read_json(Path(entry["result_path"]))
        if record["sample_id"] != sample_id or record["input_sha256"] != entry["input_sha256"]:
            raise ValueError("runと結果ファイルの対応が不正です")
        speech = record.get("speech_separation")
        reference = speech["reference_text"] if speech is not None else None
        if speech is not None and speech != separate_speech(record["transcript"], reference):
            raise ValueError("保存済みの発言分離と現在の分離処理が一致しません。同じ文字起こしから再採点してください")
        verified = validate_response(record["transcript"], record["model_response"], device_reference=reference).to_dict()
        if record["evaluation"] != verified:
            raise ValueError("採点結果と現在の検証器が一致しません。同じキャッシュから再採点してください")
        results[sample_id] = record
    return results


def character_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    reference = re.sub(r"\s+", "", normalize(reference))
    hypothesis = re.sub(r"\s+", "", normalize(hypothesis))
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1], len(reference)


def _metrics(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {**counts, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def report(manifest: dict, run: dict, *, reference_run: dict | None = None, split="validation") -> dict:
    samples = [s for s in validate_manifest(manifest) if split == "all" or s["split"] == split]
    results = _load_run(run)
    references = _load_run(reference_run) if reference_run else {}
    counts = [{"tp": 0, "fp": 0, "fn": 0, "tn": 0, "uncertain": 0} for _ in CRITERIA]
    errors, scores, coverage = [], [], []
    edit_count = reference_chars = 0
    cer_samples = 0
    reference_errors = 0
    reference_items = 0
    for sample in samples:
        sid = sample["sample_id"]
        gold = sample["labels"]
        record = results.get(sid)
        labeled = sum(v is not None for v in gold.values())
        coverage.append({"sample_id": sid, "labeled_items": labeled, "evaluated": record is not None})
        if record is None:
            continue
        if record["source_kind"] != "audio" or record["input_sha256"] != sample["audio_sha256"]:
            raise ValueError(f"音声のハッシュまたは入力種別が正解データと一致しません: {sid}")
        reference = references.get(sid)
        if reference is not None:
            if (not sample.get("reference_text") or reference["source_kind"] != "reference"
                    or reference["transcript_sha256"] != digest(sample["reference_text"].encode())):
                raise ValueError(f"人が確認した逐語録とreference runが一致しません: {sid}")
        if sample.get("reference_text"):
            edits, length = character_errors(sample["reference_text"], record["transcript"])
            edit_count += edits
            reference_chars += length
            cer_samples += 1
        for item in record["evaluation"]["items"]:
            i = item["id"]
            truth = gold[str(i)]
            if truth is None:
                continue
            predicted = item["status"] == "met"
            key = "tp" if truth and predicted else "fn" if truth else "fp" if predicted else "tn"
            counts[i - 1][key] += 1
            counts[i - 1]["uncertain"] += item["status"] == "uncertain"
            ref_correct = None
            if reference is not None:
                ref_predicted = reference["evaluation"]["items"][i - 1]["status"] == "met"
                ref_correct = ref_predicted == truth
                reference_items += 1
                reference_errors += not ref_correct
            if predicted != truth:
                errors.append({"sample_id": sid, "id": i, "gold": truth, "status": item["status"],
                               "reason": item["reason"], "warnings": item["warnings"],
                               "stage_hint": "unresolved" if ref_correct is None else
                                   "asr_or_asr_context" if ref_correct else "evaluator_also_failed"})
        if labeled == 18:
            expected = round(sum(gold.values()) / 18 * 100, 1)
            actual = record["evaluation"]["score"]
            scores.append({"sample_id": sid, "expected": expected, "actual": actual,
                           "absolute_error": abs(actual - expected)})
    totals = {key: sum(c[key] for c in counts) for key in counts[0]}
    n = sum(totals[k] for k in ("tp", "fp", "fn", "tn"))
    if not n:
        raise ValueError("比較できる注釈済み項目がありません。未注釈を未実施として補完しません")
    return {"created_at": now_iso(), "split": split, "run_status": run["status"],
            "coverage": coverage, "evaluated_labeled_items": n,
            "micro": _metrics(totals), "uncertain_rate": totals["uncertain"] / n,
            "per_item": [{"id": i + 1, "name": CRITERIA[i][0], **_metrics(c)} for i, c in enumerate(counts)],
            "scores": scores, "score_mae": sum(s["absolute_error"] for s in scores) / len(scores) if scores else None,
            "cer": edit_count / reference_chars if reference_chars else None,
            "cer_samples": cer_samples, "reference_evaluated_items": reference_items,
            "reference_evaluator_errors": reference_errors if reference_items else None,
            "errors": errors,
            "limitations": "同一収録日の初期検証。未注釈・未処理は分母に入れずcoverageに表示。stage_hintは切り分けの手掛かりで、ASR原因の断定ではない。"}


def run_references(manifest: dict, pipeline, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError("既存の実行記録は上書きしません。新しい保存先を指定してください")
    samples = [s for s in validate_manifest(manifest) if s.get("reference_text")]
    if not samples:
        raise ValueError("人が確認したreference_textがまだありません")
    run = {"created_at": now_iso(), "status": "running", "stage": "reference", "results": [], "errors": []}
    write_json(destination, run)
    for sample in samples:
        try:
            text = sample["reference_text"]
            document = pipeline.evaluate_text(text)
            result = pipeline.export(document, sample_id=sample["sample_id"], source_path="gold.reference_text",
                                     input_sha256=digest(text.encode()), source_kind="reference")
            run["results"].append({k: result[k] for k in ("sample_id", "input_sha256", "result_path")})
        except Exception as exc:
            run["status"] = "interrupted"
            run["errors"].append({"sample_id": sample["sample_id"], "type": type(exc).__name__, "message": str(exc)})
            write_json(destination, run)
            return run
        write_json(destination, run)
    run["status"] = "completed"
    write_json(destination, run)
    return run
