"""Local score summaries against human labels. Gold data never enters a model prompt."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

from src.bls_evaluation import CRITERIA, normalize

HUMAN_SCORES_CSV = Path(__file__).resolve().parents[1] / "data" / "人間音声採点データ.csv"


def _sample_id(value: str) -> str:
    return normalize(Path(value.strip()).stem).strip()


def _natural_key(value: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def load_human_scores(path: Path | None = HUMAN_SCORES_CSV) -> dict:
    if path is None:
        return {"path": None, "sha256": None, "samples": {}, "note": "模範CSV未指定のため比較なし"}
    path = Path(path).resolve()
    if path == HUMAN_SCORES_CSV and not path.exists():
        return {"path": str(path), "sha256": None, "samples": {}, "note": "模範CSVが見つからないため比較なし"}
    raw = path.read_bytes()
    rows = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    header = next(rows, [])
    if (len(header) != 21 or header[0].strip() != "BLS評価対象コール"
            or [v.strip() for v in header[-2:]] != ["成功項目数", "点数"]):
        raise ValueError("模範CSVは音声名・18評価項目（1〜18の順）・成功項目数・点数の21列が必要です")
    samples = {}
    for number, row in enumerate(rows, 2):
        if not any(v.strip() for v in row):
            continue
        if len(row) != 21:
            raise ValueError(f"模範CSVの{number}行目の列数が不正です")
        sid = _sample_id(row[0])
        if not sid or sid in samples:
            raise ValueError(f"模範CSVの{number}行目の音声名が空または重複しています: {sid}")
        values = [normalize(v).strip() for v in row[1:19]]
        if any(v not in ("0", "1") for v in values):
            raise ValueError(f"模範CSVの{number}行目の18項目は0か1で指定してください")
        labels = [v == "1" for v in values]
        try:
            met_count = int(row[19])
            score = float(row[20])
        except ValueError as exc:
            raise ValueError(f"模範CSVの{number}行目の成功項目数または点数が数値ではありません") from exc
        if (met_count != sum(labels) or not math.isfinite(score) or not 0 <= score <= 100
                or round(score, 1) != round(met_count / 18 * 100, 1)):
            raise ValueError(f"模範CSVの{number}行目で18項目・成功項目数・点数が一致しません")
        samples[sid] = {"score": round(score, 1), "labels": labels}
    if not samples:
        raise ValueError("模範CSVに採点データがありません")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "samples": samples, "note": ""}


def _load_result(entry: dict, results_dir: Path | None) -> dict:
    expected = Path(entry["result_path"])
    if results_dir is not None or not expected.is_file():
        # Moving results into revised3/ is supported, but never guess by sample ID:
        # the full filename identifies the exact input, prompt and validation run.
        directory = Path(results_dir) if results_dir is not None else expected.parent
        matches = list(directory.rglob(expected.name))
        if len(matches) != 1:
            raise ValueError(f"結果JSONを一意に特定できません: {expected.name}（候補{len(matches)}件）")
        expected = matches[0]
    record = json.loads(expected.read_text(encoding="utf-8"))
    if record["sample_id"] != entry["sample_id"] or record["input_sha256"] != entry["input_sha256"]:
        raise ValueError(f"runと結果JSONの対応が不正です: {expected}")
    return record


def render_score_summary(run: dict, gold: dict, *, results_dir: Path | None = None) -> str:
    if run.get("stage") == "transcription":
        raise ValueError("文字起こし専用runには採点結果がありません")
    entries = run["results"]
    if len({e["result_path"] for e in entries}) != len(entries):
        raise ValueError("runに同じ結果JSONが重複しています")
    records = [_load_result(entry, results_dir) for entry in sorted(entries, key=lambda e: _natural_key(e["sample_id"]))]
    rows, differences, mismatches = [], [], []
    actuals, ideals = [], []
    models, vocabularies = set(), set()
    for record in records:
        sid = record["sample_id"]
        ev = record["evaluation"]
        score = round(ev["score"], 1)
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError(f"Geminiの点数が不正です: {sid}")
        models.add(record.get("evaluation_model", "不明"))
        transcription = record.get("transcription") or {}
        config = transcription.get("settings", {}).get("config", {})
        if "custom_vocabulary" in config:
            vocabularies.add(tuple(config["custom_vocabulary"]))
        human = gold["samples"].get(_sample_id(sid))
        if human is None:
            rows.append(f"{sid}: Gemini={score:.1f}点 / 模範=なし / 差=比較不可 / 判定不能={ev['uncertain_count']}項目")
            continue
        delta = round(score - human["score"], 1)
        rows.append(f"{sid}: Gemini={score:.1f}点 / 模範={human['score']:.1f}点 / 差={delta:+.1f}点 / 絶対差={abs(delta):.1f}点 / 判定不能={ev['uncertain_count']}項目")
        actuals.append(score)
        ideals.append(human["score"])
        differences.append(delta)
        item_differences = []
        for item in sorted(ev["items"], key=lambda item: item["id"]):
            i = item["id"]
            truth = human["labels"][i - 1]
            predicted = item["status"] == "met"
            # An uncertain negative is still shown, even if it has the same zero points.
            if truth != predicted or item["status"] == "uncertain":
                ideal = "達成" if truth else "未達成"
                status = {"met": "達成", "not_detected": "未検出", "uncertain": "判定不能"}[item["status"]]
                item_differences.append(f"{i}.{CRITERIA[i - 1][0]}（模範:{ideal} / Gemini:{status}）")
        if item_differences:
            mismatches.append(f"{sid}: " + "、".join(item_differences))

    lines = ["BLS採点スコアと模範解答との差", f"実行日時: {run.get('created_at', '不明')}",
             f"実行記録: {run.get('run_path', '不明')}", f"実行状態: {run['status']}",
             "採点モデル: " + (", ".join(sorted(models)) or "採点結果なし"),
             f"語彙設定: {run.get('vocabulary', 'runに記録なし')}（結果内のカスタム語彙: {len(vocabularies)}種類）",
             f"模範CSV: {gold['path'] or '未指定'}", f"模範CSV SHA-256: {gold['sha256'] or 'なし'}",
             f"採点済み: {len(records)}本 / 比較可能: {len(differences)}本 / 模範なし: {len(records) - len(differences)}本",
             "差 = Gemini点数 − 模範点数（負: 過小評価、正: 過大評価）。点数を小数1桁に丸めて比較。",
             "模範CSVは集計だけに使用し、Geminiの採点には渡していません。"]
    if gold["note"]:
        lines.append(gold["note"])
    if len({e["sample_id"] for e in entries}) != len(entries):
        lines.append("同じ音声名の結果が複数あります。音声・動画などの入力ファイルごとに集計しています。")
    lines += ["", "音声別スコア（番号順）", *rows]
    if differences:
        n = len(differences)
        lines += ["", "全体の乖離（比較可能な音声のみ）",
                  f"平均点: Gemini={sum(actuals) / n:.1f}点 / 模範={sum(ideals) / n:.1f}点",
                  f"平均差: {sum(differences) / n:+.1f}点",
                  f"平均絶対誤差（MAE）: {sum(abs(d) for d in differences) / n:.1f}点",
                  f"最大絶対誤差: {max(abs(d) for d in differences):.1f}点",
                  f"点数一致: {sum(d == 0 for d in differences)}/{n}本（項目ごとの一致とは異なります）"]
    else:
        lines += ["", "全体の乖離: 比較対象がないため算出不可"]
    lines += ["", "不一致・判定不能の項目", *(mismatches or ["なし" if differences else "比較不可"])]
    incomplete = {p: "未処理" for p in run.get("pending", [])}
    for error in run.get("errors", []):
        incomplete[error.get("source_path", error.get("sample_id", "不明"))] = f"エラー: {error['type']}"
    if incomplete:
        lines += ["", "未採点（0点として扱わず、乖離の集計から除外）"]
        lines += [f"{Path(path).stem}: {state}" for path, state in sorted(incomplete.items(), key=lambda pair: _natural_key(pair[0]))]
    return "\n".join(lines) + "\n"
