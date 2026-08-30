"""Validation helpers for BLS call scorer research."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sklearn.metrics import cohen_kappa_score, f1_score

from src.bls_scoring import load_rubric


ANNOTATION_FIELDS = [
    "session_id",
    "sample_id",
    "split",
    "source_type",
    "annotator_id",
    "annotator_role",
    "record_type",
    "item_id",
    "label",
    "score",
    "evidence",
    "notes",
]


def read_csv(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def create_annotation_template(
    manifest_path: Path | str,
    output_path: Path | str,
    *,
    rubric_path: Path | str,
) -> int:
    manifest = read_csv(manifest_path)
    required = {"session_id", "sample_id", "transcript_path", "audio_path", "split"}
    if not manifest or not required.issubset(manifest[0]):
        raise ValueError(f"manifestには次の列が必要です: {sorted(required)}")
    session_splits: dict[str, set[str]] = defaultdict(set)
    for sample in manifest:
        session_splits[sample["session_id"]].add(sample["split"])
    mixed_sessions = sorted(
        session_id for session_id, splits in session_splits.items() if len(splits) > 1
    )
    if mixed_sessions:
        raise ValueError(
            "同一session_idを複数splitへ配置できません: " + ", ".join(mixed_sessions)
        )
    rubric = load_rubric(rubric_path)
    call_items = [("call", item["id"], item["label"]) for item in rubric["calls"]]
    sequence_items = [
        ("sequence", item["id"], item["label"])
        for item in rubric["sequence_constraints"]
    ]
    rows: list[dict[str, Any]] = []
    seen_audio_sessions: set[str] = set()
    for sample in manifest:
        sources = [("text", sample["sample_id"])]
        if sample["session_id"] not in seen_audio_sessions:
            sources.append(("audio", sample["sample_id"]))
            seen_audio_sessions.add(sample["session_id"])
        for source_type, sample_id in sources:
            for role in ("expert", "nonexpert"):
                for record_type, item_id, label in call_items + sequence_items:
                    rows.append(
                        {
                            "session_id": sample["session_id"],
                            "sample_id": sample_id,
                            "split": sample["split"],
                            "source_type": source_type,
                            "annotator_id": "",
                            "annotator_role": role,
                            "record_type": record_type,
                            "item_id": item_id,
                            "label": label,
                            "score": "",
                            "evidence": "",
                            "notes": "",
                        }
                    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _annotation_key(row: Mapping[str, str]) -> tuple[str, str, str, str]:
    subject_id = row["sample_id"] if row["source_type"] == "text" else row["session_id"]
    return row["source_type"], subject_id, row["record_type"], row["item_id"]


def _parse_score(row: Mapping[str, str]) -> float | None:
    raw = row.get("score", "").strip()
    if not raw:
        return None
    value = float(raw)
    allowed = {0.0, 0.5, 1.0} if row["record_type"] == "call" else {0.0, 1.0}
    if value not in allowed:
        raise ValueError(
            f"不正な注釈スコアです: {row['item_id']}={value}（許容値: {sorted(allowed)}）"
        )
    return value


def _ground_truth(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str, str], float]:
    grouped: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        score = _parse_score(row)
        if score is not None:
            grouped[_annotation_key(row)][row["annotator_role"]].append(score)
    truth: dict[tuple[str, str, str, str], float] = {}
    for key, by_role in grouped.items():
        preferred = by_role.get("adjudicated") or by_role.get("expert")
        if preferred:
            truth[key] = preferred[-1]
    return truth


def _interrater(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        score = _parse_score(row)
        if score is not None and row["annotator_role"] in {"expert", "nonexpert"}:
            grouped[_annotation_key(row)][row["annotator_role"]] = score
    pairs = [values for values in grouped.values() if {"expert", "nonexpert"}.issubset(values)]
    if not pairs:
        return {"n": 0, "weighted_kappa": None, "exact_agreement": None}
    expert = [pair["expert"] for pair in pairs]
    nonexpert = [pair["nonexpert"] for pair in pairs]
    expert_classes = [int(value * 2) for value in expert]
    nonexpert_classes = [int(value * 2) for value in nonexpert]
    kappa = (
        cohen_kappa_score(expert_classes, nonexpert_classes, weights="quadratic")
        if len(set(expert_classes) | set(nonexpert_classes)) > 1
        else math.nan
    )
    return {
        "n": len(pairs),
        "weighted_kappa": None if math.isnan(kappa) else round(float(kappa), 4),
        "exact_agreement": round(sum(a == b for a, b in zip(expert, nonexpert)) / len(pairs), 4),
    }


def _prediction_rows(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_like in paths:
        path = Path(path_like)
        bundle = json.loads(path.read_text(encoding="utf-8"))
        sample_id = bundle.get("sample_id") or Path(bundle["source"]).stem
        session_id = bundle.get("session_id") or sample_id
        for result in bundle["results"]:
            base = {
                "sample_id": sample_id,
                "session_id": session_id,
                "method": result["method"],
                "prediction_file": str(path),
            }
            for call in result["calls"]:
                rows.append({**base, "record_type": "call", "item_id": call["call_id"], "score": float(call["score"])})
            for sequence in result["sequences"]:
                if sequence["status"] != "not_evaluable":
                    rows.append(
                        {
                            **base,
                            "record_type": "sequence",
                            "item_id": sequence["constraint_id"],
                            "score": 1.0 if sequence["status"] == "pass" else 0.0,
                        }
                    )
    return rows


def _metric_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    calls = [row for row in rows if row["record_type"] == "call"]
    sequences = [row for row in rows if row["record_type"] == "sequence"]
    if calls:
        truth = [row["truth"] for row in calls]
        prediction = [row["prediction"] for row in calls]
        truth_classes = [int(value * 2) for value in truth]
        prediction_classes = [int(value * 2) for value in prediction]
        exact = sum(a == b for a, b in zip(truth, prediction)) / len(calls)
        macro_f1 = f1_score(
            truth_classes,
            prediction_classes,
            labels=sorted(set(truth_classes) | set(prediction_classes)),
            average="macro",
            zero_division=0,
        )
        kappa = (
            cohen_kappa_score(truth_classes, prediction_classes, weights="quadratic")
            if len(set(truth_classes) | set(prediction_classes)) > 1
            else math.nan
        )
        grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"truth": [], "prediction": []})
        for row in calls:
            key = row["session_id"], row["sample_id"]
            grouped[key]["truth"].append(row["truth"])
            grouped[key]["prediction"].append(row["prediction"])
        content_mae = statistics.mean(
            abs(statistics.mean(values["truth"]) - statistics.mean(values["prediction"])) * 100
            for values in grouped.values()
        )
    else:
        exact = macro_f1 = kappa = content_mae = None
    sequence_accuracy = (
        sum(row["truth"] == row["prediction"] for row in sequences) / len(sequences)
        if sequences
        else None
    )
    return {
        "call_exact_accuracy": exact,
        "call_macro_f1": macro_f1,
        "call_weighted_kappa": None if kappa is None or math.isnan(kappa) else float(kappa),
        "content_score_mae": content_mae,
        "sequence_accuracy": sequence_accuracy,
    }


def _bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float] | None]:
    if samples < 1 or not rows:
        return {}
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[row["session_id"]].append(row)
    session_ids = sorted(by_session)
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        sampled_rows: list[Mapping[str, Any]] = []
        for session_id in rng.choices(session_ids, k=len(session_ids)):
            sampled_rows.extend(by_session[session_id])
        for name, value in _metric_values(sampled_rows).items():
            if value is not None:
                distributions[name].append(value)
    intervals: dict[str, list[float] | None] = {}
    for name, values in distributions.items():
        values.sort()
        low = values[int(0.025 * (len(values) - 1))]
        high = values[int(0.975 * (len(values) - 1))]
        intervals[name] = [round(low, 4), round(high, 4)]
    return intervals


def validate_predictions(
    annotation_path: Path | str,
    prediction_paths: Iterable[Path | str],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 20260830,
    split: str | None = "test",
) -> dict[str, Any]:
    annotations = read_csv(annotation_path)
    if split is not None:
        annotations = [
            row for row in annotations if not row.get("split", "").strip() or row["split"] == split
        ]
    truth = _ground_truth(annotations)
    predictions = _prediction_rows(prediction_paths)
    comparisons: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        for source_type in ("text", "audio"):
            subject_id = prediction["sample_id"] if source_type == "text" else prediction["session_id"]
            key = (source_type, subject_id, prediction["record_type"], prediction["item_id"])
            if key in truth:
                comparisons[(prediction["method"], source_type)].append(
                    {
                        **prediction,
                        "truth": truth[key],
                        "prediction": prediction["score"],
                    }
                )
    result_groups = []
    for (method, source_type), rows in sorted(comparisons.items()):
        metrics = _metric_values(rows)
        result_groups.append(
            {
                "method": method,
                "reference": source_type,
                "n_rows": len(rows),
                "metrics": {name: None if value is None else round(value, 4) for name, value in metrics.items()},
                "bootstrap_95_ci": _bootstrap_ci(rows, samples=bootstrap_samples, seed=seed),
            }
        )
    return {
        "interrater": _interrater(annotations),
        "comparisons": result_groups,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "split": split,
    }
