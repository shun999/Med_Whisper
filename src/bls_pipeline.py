"""Gemini adapters, durable request budgets, stage caches, and result exports."""

import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid
import warnings
from zoneinfo import ZoneInfo

from src.bls_evaluation import (
    BASELINE_VOCABULARY, REVISED_VOCABULARY, PROMPT_VERSION, RUBRIC_VERSION,
    SYSTEM_INSTRUCTION, RESPONSE_SCHEMA, VALIDATOR_VERSION, InvalidResponse,
    build_prompt, clean_vocabulary, validate_response,
)

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIBE_MODEL = "gemini-3.5-transcribe"
EVALUATION_MODEL = "gemini-3.8-flash"
TRANSCRIPTION_RPM = 2
TRANSCRIPTION_RPD = 100
EVALUATION_RPM = 1000
EVALUATION_RPD = 10_000
MIME_TYPES = {".wav": "audio/wav", ".mp3": "audio/mp3", ".flac": "audio/flac",
              ".m4a": "audio/m4a", ".ogg": "audio/ogg"}
VIDEO_TYPES = {".mp4", ".mov"}
PACIFIC = ZoneInfo("America/Los_Angeles")


def digest(value) -> str:
    content = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_output(path: Path) -> Path:
    path = path.resolve()
    allowed = ((ROOT / "outputs").resolve(), Path(tempfile.gettempdir()).resolve())
    if not any(path.is_relative_to(base) and path != base for base in allowed):
        raise ValueError("生成先はプロジェクトのoutputs/配下または一時ディレクトリ配下にしてください")
    # A symlink from outputs into protected directories has already been resolved.
    if any(path.is_relative_to((ROOT / name).resolve()) for name in ("data", "env", ".venv", ".git", ".agents", ".codex")):
        raise ValueError("入力データ・仮想環境・リポジトリ管理領域には書き込めません")
    return path


def atomic_text(path: Path, text: str) -> None:
    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class QuotaExhausted(RuntimeError):
    pass


class MissingAPIKey(RuntimeError):
    pass


def create_gemini_client(api_key: str, *, httpx_client=None):
    """Disable SDK retries: every inference attempt must pass our budget first."""
    from google import genai
    options = {"retry_options": {"attempts": 1}, "timeout": 300_000}
    if httpx_client is not None:
        options["httpx_client"] = httpx_client
    client = genai.Client(api_key=api_key, http_options=options)
    # google-genai 2.20 clamps attempts=0 to 1, then the Interactions adapter
    # interprets 1 as one *retry*. Explicitly disable its generated retry strategy.
    client.interactions.sdk_configuration.retry_config.strategy = "none"
    return client


class RequestBudget:
    """One lock per project scope; multiple processes share admission timestamps.

    Counts attempted inference requests (including failures), not file uploads.
    Other applications' usage cannot be inferred; server 429 remains authoritative.
    """

    def __init__(self, directory: Path | None = None, *, scope="default", clock=time.time, sleep=time.sleep):
        self.directory = safe_output(directory or ROOT / "outputs" / ".bls_state")
        self.scope = digest(scope)[:24]
        self.clock, self.sleep = clock, sleep

    def acquire(self, model: str, rpm: int, rpd: int) -> None:
        if rpm <= 0 or rpd <= 0:
            raise ValueError("API上限は正の整数で指定してください")
        self.directory.mkdir(parents=True, exist_ok=True)
        state_path = self.directory / f"{self.scope}.json"
        while True:
            with (self.directory / f"{self.scope}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                state = read_json(state_path) if state_path.exists() else {}
                now = self.clock()
                today = datetime.fromtimestamp(now, PACIFIC).date()
                stamps = [t for t in state.get(model, [])
                          if datetime.fromtimestamp(t, PACIFIC).date() == today or t > now - 60]
                daily = [t for t in stamps if datetime.fromtimestamp(t, PACIFIC).date() == today]
                if len(daily) >= rpd:
                    raise QuotaExhausted(f"{model}: 日上限{rpd}回に到達。米国太平洋時間の翌日以降に同じコマンドで再開できます")
                recent = sorted(t for t in stamps if t > now - 60)
                if len(recent) < rpm:
                    state[model] = stamps + [now]
                    write_json(state_path, state)
                    return
                delay = max(0.01, recent[-rpm] + 60.01 - now)
            self.sleep(min(delay, 60))


class BLSPipeline:
    def __init__(self, *, client=None, output_dir: Path | None = None, budget=None,
                 transcription_model=TRANSCRIBE_MODEL, evaluation_model=EVALUATION_MODEL,
                 vocabulary="revised", transcription_rpm=TRANSCRIPTION_RPM,
                 transcription_rpd=TRANSCRIPTION_RPD, evaluation_rpm=EVALUATION_RPM,
                 evaluation_rpd=EVALUATION_RPD, quota_scope="default", sleep=time.sleep):
        self.output_dir = safe_output(output_dir or ROOT / "outputs" / "evaluation" / "bls")
        self.client = client
        self.budget = budget or RequestBudget(scope=quota_scope)
        self.transcription_model, self.evaluation_model = transcription_model, evaluation_model
        profiles = {"baseline": BASELINE_VOCABULARY, "revised": REVISED_VOCABULARY}
        if isinstance(vocabulary, str):
            if vocabulary not in profiles:
                raise ValueError("語彙はbaseline/revisedまたは文字列リストで指定してください")
            self.vocabulary_name, terms = vocabulary, profiles[vocabulary]
        else:
            self.vocabulary_name, terms = "custom", vocabulary
        self.vocabulary = clean_vocabulary(terms)
        self.limits = {"transcription": (transcription_rpm, transcription_rpd),
                       "evaluation": (evaluation_rpm, evaluation_rpd)}
        if any(type(n) is not int or n <= 0 for pair in self.limits.values() for n in pair):
            raise ValueError("API上限は正の整数で指定してください")
        self.sleep = sleep

    def _client(self):
        if self.client is None:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise MissingAPIKey("APIキーがありません。GEMINI_API_KEYを設定するかノートブックの非表示入力を使ってください")
            self.client = create_gemini_client(key)
        config = getattr(self.client.interactions, "sdk_configuration", None)
        if config is not None and getattr(config.retry_config, "strategy", None) != "none":
            raise ValueError("SDKの内部再試行を無効にする必要があります。create_gemini_client()でクライアントを作成してください")
        return self.client

    def _request(self, stage, **kwargs):
        client = self._client()  # Missing credentials must not consume quota.
        for attempt in range(3):
            self.budget.acquire(kwargs["model"], *self.limits[stage])
            try:
                response = client.interactions.create(**kwargs)
                status = getattr(response, "status", "completed")
                if status not in (None, "completed"):
                    raise InvalidResponse(f"API応答が完了していません: {status}")
                if not getattr(response, "output_text", "").strip():
                    raise InvalidResponse("APIから空の応答が返されました")
                return response
            except Exception as exc:
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if str(code) not in {"429", "500", "502", "503", "504"} or attempt == 2:
                    raise
                headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
                try:
                    delay = max(float(headers.get("Retry-After", 0)), 2 ** attempt * 5)
                except (ValueError, TypeError):
                    delay = 2 ** attempt * 5
                # Do not hold a quota lock during network calls or backoff.
                while delay > 0:
                    step = min(delay, 60)
                    self.sleep(step)
                    delay -= step
        raise AssertionError("unreachable")

    @staticmethod
    def _raw(response):
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return {"output_text": response.output_text}

    def evaluate_text(self, text: str) -> dict:
        prompt = build_prompt(text)
        settings = {"model": self.evaluation_model, "rubric_version": RUBRIC_VERSION,
                    "prompt_version": PROMPT_VERSION, "system_instruction": SYSTEM_INSTRUCTION,
                    "prompt": prompt, "response_schema": RESPONSE_SCHEMA}
        key = digest(settings)
        response_path = self.output_dir / "cache" / "evaluation" / f"{key}.json"
        if response_path.exists():
            cached = read_json(response_path)
            if cached["key"] != key or cached["settings"] != settings:
                raise InvalidResponse("採点キャッシュの設定が一致しません")
        else:
            response = self._request("evaluation", model=self.evaluation_model,
                system_instruction=SYSTEM_INSTRUCTION, input=prompt,
                response_format={"type": "text", "mime_type": "application/json", "schema": RESPONSE_SCHEMA})
            # Preserve the raw response even if JSON parsing/validation fails.
            cached = {"key": key, "created_at": now_iso(), "settings": settings,
                      "raw_response": self._raw(response), "output_text": response.output_text}
            write_json(self.output_dir / "responses" / f"{key}_{uuid.uuid4().hex}.json", cached)
        try:
            payload = json.loads(cached["output_text"])
        except (ValueError, TypeError) as exc:
            raise InvalidResponse(f"採点応答がJSONではありません。応答保存先: {response_path}") from exc
        result = validate_response(text, payload)
        if not response_path.exists():
            write_json(response_path, cached)
        return {"evaluation": result.to_dict(), "model_response": payload, "transcript": text,
                "transcript_sha256": digest(text.encode()), "evaluation_model": self.evaluation_model,
                "cache_key": key, "response_path": str(response_path), "created_at": cached["created_at"]}

    def transcribe(self, path: Path) -> dict:
        path = Path(path).resolve()
        if path.suffix.lower() not in MIME_TYPES.keys() | VIDEO_TYPES:
            raise ValueError(f"未対応の音声/動画形式: {path.suffix}")
        sha = file_digest(path)
        config = {"language_codes": ["ja-JP"], "custom_vocabulary": self.vocabulary,
                  "mode": {"type": "verbatim"}}
        settings = {"input_sha256": sha, "model": self.transcription_model, "config": config,
                    "conversion": "ffmpeg-mono-16k-pcm-v1" if path.suffix.lower() in VIDEO_TYPES else "original"}
        key = digest(settings)
        cache_path = self.output_dir / "cache" / "transcription" / f"{key}.json"
        if cache_path.exists():
            cached = read_json(cache_path)
            if cached["settings"] != settings or not cached.get("text", "").strip():
                raise InvalidResponse("文字起こしキャッシュが不正です")
            return cached
        client = self._client()
        uploaded = None
        try:
            with tempfile.TemporaryDirectory(prefix="medwhisper_bls_") as temp:
                if path.suffix.lower() in VIDEO_TYPES:
                    upload_path = Path(temp) / "audio.wav"
                    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn",
                                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(upload_path)],
                                   check=True, capture_output=True)
                else:
                    upload_path = Path(temp) / f"audio{path.suffix.lower()}"
                    shutil.copyfile(path, upload_path)
                uploaded = client.files.upload(file=upload_path, config={"mime_type": MIME_TYPES[upload_path.suffix]})
                response = self._request("transcription", model=self.transcription_model,
                    input=[{"type": "audio", "uri": uploaded.uri, "mime_type": uploaded.mime_type}],
                    generation_config={"transcription_config": config})
        finally:
            if uploaded is not None and getattr(uploaded, "name", None):
                try:
                    client.files.delete(name=uploaded.name)
                except Exception:
                    warnings.warn("Gemini上の一時ファイルを削除できませんでした。Files APIで確認してください")
        cached = {"key": key, "settings": settings, "source_path": str(path), "input_sha256": sha,
                  "created_at": now_iso(), "text": response.output_text.strip(), "raw_response": self._raw(response)}
        write_json(cache_path, cached)
        atomic_text(cache_path.with_suffix(".txt"), cached["text"] + "\n")
        return cached

    def export(self, document: dict, *, sample_id: str, source_path: str, input_sha256: str,
               source_kind: str, transcription: dict | None = None) -> dict:
        record = {**document, "sample_id": sample_id, "source_path": source_path,
                  "source_kind": source_kind, "input_sha256": input_sha256,
                  "transcription": transcription, "exported_at": now_iso()}
        key = digest({"input": input_sha256, "evaluation": document["cache_key"],
                      "validator": VALIDATOR_VERSION, "sample_id": sample_id,
                      "source_path": source_path, "source_kind": source_kind})
        stem = re_safe_stem(sample_id)
        path = self.output_dir / "results" / f"{stem}_{key}.json"
        if not path.exists():
            write_json(path, record)
        record = read_json(path)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=["id", "name", "status", "reason", "evidence", "warnings"])
        writer.writeheader()
        for row in record["evaluation"]["items"]:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], list) else row[k]
                             for k in writer.fieldnames})
        atomic_text(path.with_suffix(".csv"), "\ufeff" + stream.getvalue())
        return {**record, "result_path": str(path)}

    def evaluate_file(self, path: Path) -> dict:
        path = Path(path).resolve()
        is_text = path.suffix.lower() == ".txt"
        transcription = None if is_text else self.transcribe(path)
        text = path.read_text(encoding="utf-8") if is_text else transcription["text"]
        document = self.evaluate_text(text)
        return self.export(document, sample_id=path.stem, source_path=str(path),
                           input_sha256=file_digest(path), source_kind="text" if is_text else "audio",
                           transcription=transcription)

    def run(self, paths: list[Path], *, transcription_only=False) -> dict:
        if not paths:
            raise ValueError("対象ファイルがありません")
        # Validate before the first external call; preserve unprocessed paths on interruption.
        resolved = [Path(p).resolve() for p in paths]
        if len(set(resolved)) != len(resolved):
            raise ValueError("入力ファイルが重複しています")
        for path in resolved:
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix.lower() not in MIME_TYPES.keys() | VIDEO_TYPES | {".txt"}:
                raise ValueError(f"未対応の入力: {path}")
            if transcription_only and path.suffix.lower() == ".txt":
                raise ValueError("TXT入力に文字起こし専用モードは使えません")
        run_path = self.output_dir / "runs" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:12]}.json"
        run = {"created_at": now_iso(), "status": "running", "run_path": str(run_path),
               "stage": "transcription" if transcription_only else "evaluation",
               "inputs": [str(p) for p in resolved], "results": [], "errors": [], "pending": [str(p) for p in resolved]}
        write_json(run_path, run)
        for path in resolved:
            try:
                if transcription_only:
                    transcript = self.transcribe(path)
                    result = {"sample_id": path.stem, "input_sha256": transcript["input_sha256"],
                              "result_path": str(self.output_dir / "cache" / "transcription" / f"{transcript['key']}.json")}
                else:
                    result = self.evaluate_file(path)
                run["results"].append({"sample_id": result["sample_id"], "input_sha256": result["input_sha256"],
                                       "result_path": result["result_path"]})
            except (QuotaExhausted, MissingAPIKey) as exc:
                run["status"] = "interrupted"
                run["errors"].append({"source_path": str(path), "type": type(exc).__name__, "message": str(exc)})
                write_json(run_path, run)
                return run
            except Exception as exc:
                run["errors"].append({"source_path": str(path), "type": type(exc).__name__, "message": str(exc)})
            run["pending"].remove(str(path))
            write_json(run_path, run)
        run["status"] = "completed" if not run["errors"] else "failed"
        write_json(run_path, run)
        return run


def re_safe_stem(stem: str) -> str:
    import re
    return re.sub(r"[^\w.-]", "_", stem)[:80] or "sample"
