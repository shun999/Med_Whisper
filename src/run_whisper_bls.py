# src/run_whisper_bls.py
from __future__ import annotations

import os
from typing import List, Optional, Dict, Any

from whisper import load_model
from whisper.tokenizer import get_tokenizer
from whisper.utils import format_timestamp

# ---- beam hook ----
# InspectConfig / save_inspect_log_jsonl が beam_hook に無い場合は、次に beam_hook を直す必要があります
from src.beam_hook import (
    BiasConfig,
    LogConfig,
    InspectConfig,              # ★追加
    install_beam_hook,
    uninstall_beam_hook,
    save_beam_log_jsonl,
    save_inspect_log_jsonl,     # ★追加
)


# def save_srt(result: Dict[str, Any], srt_path: str) -> None:
#     with open(srt_path, "w", encoding="utf-8") as f:
#         for i, seg in enumerate(result.get("segments", []), start=1):
#             s = format_timestamp(seg["start"], always_include_hours=True, decimal_marker=",")
#             e = format_timestamp(seg["end"],   always_include_hours=True, decimal_marker=",")
#             f.write(f"{i}\n{s} --> {e}\n{seg['text'].strip()}\n\n")
def save_srt(result: Dict[str, Any], srt_path: str) -> None:
    def _clamp_ts(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        # まれに -1e-6 みたいなのが出るので 0 に丸める
        if v < 0.0:
            v = 0.0
        return v

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result.get("segments", []), start=1):
            s = _clamp_ts(seg.get("start", 0.0))
            e = _clamp_ts(seg.get("end", 0.0))

            # end < start もまれに起きるので補正
            if e < s:
                e = s

            ss = format_timestamp(s, always_include_hours=True, decimal_marker=",")
            ee = format_timestamp(e, always_include_hours=True, decimal_marker=",")
            text = (seg.get("text", "") or "").strip()

            f.write(f"{i}\n{ss} --> {ee}\n{text}\n\n")

def transcribe_with_optional_bias(
    audio_path: str,
    out_dir: str,
    domain_terms: List[str],
    initial_prompt: Optional[str] = None,
    use_bias: bool = False,
    beam_size: int = 3,
    temperature: Optional[float] = 0.0,
    # ---- NEW: inspect controls ----
    inspect_cfg: Optional[InspectConfig] = None,
    inspect_out_jsonl: str = "inspect.jsonl",
    # ---- Optional: expose bias/log configs if you want to tune from caller ----
    bias_cfg: Optional[BiasConfig] = None,
    log_cfg: Optional[LogConfig] = None,
    model_name: str = "large-v3-turbo",
) -> Dict[str, Any]:
    """
    - use_bias=True で BeamSearchDecoder.update をフックし、domain terms の next-token bias を適用
    - inspect_cfg を渡すと、各 step の top-k に「指定文字列が存在するか」を JSONL に記録する
      (例: targets=["傷","傷病者"])
    """
    os.makedirs(out_dir, exist_ok=True)

    model = load_model(model_name)
    model.eval()
    tokenizer = get_tokenizer(multilingual=True, task="transcribe")

    orig_update = None
    state = None

    # defaults
    if bias_cfg is None:
        bias_cfg = BiasConfig(
            enable_bias=True,
            bias_strength=1.2,
            max_suffix_tokens=32,
            disable_bias_if_no_prefix=True,  # safety
            clamp_bias=3.0,
        )
    if log_cfg is None:
        log_cfg = LogConfig(enable_log=True, topk=10)

    # ---- install hook ----
    # if use_bias:
    #     # InspectConfig が None の場合は inspect 無効のまま動く想定
    #     orig_update, state = install_beam_hook(
    #         tokenizer=tokenizer,
    #         domain_terms=domain_terms,
    #         bias_cfg=bias_cfg,
    #         log_cfg=log_cfg,
    #         inspect_cfg=inspect_cfg,  # ★追加
    #     )
    # ---- install hook ----
    # inspect だけしたい場合でも hook は必要
    need_hook = bool(use_bias) or (inspect_cfg is not None and inspect_cfg.enable_inspect)

    if need_hook:
        # use_bias=False のときは bias 自体を無効化（hookは入れるが加算しない）
        bias_cfg.enable_bias = bool(use_bias)

        orig_update, state = install_beam_hook(
            tokenizer=tokenizer,
            domain_terms=domain_terms,
            bias_cfg=bias_cfg,
            log_cfg=log_cfg,
            inspect_cfg=inspect_cfg,
        )
    try:
        kwargs = dict(
            language="ja",
            task="transcribe",
            beam_size=beam_size,
            condition_on_previous_text=False,
            #without_timestamps=False,  # timestamps はセグメント分割に必要だが、今回は不要なのでオフ
        )
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        result = model.transcribe(audio_path, **kwargs)

    finally:
        if orig_update is not None:
            uninstall_beam_hook(orig_update)

    # ---- outputs ----
    base = os.path.join(out_dir, "result_bias" if use_bias else "result_default")
    save_srt(result, base + ".srt")

    if state is not None:
        # existing beam log
        save_beam_log_jsonl(state, base + "_beamlog.jsonl")

        # NEW: inspect log
        # out_dir に JSONL を落とす（ファイル名は引数で変更可能）
        try:
            save_inspect_log_jsonl(state, os.path.join(out_dir, inspect_out_jsonl))
        except Exception:
            # InspectConfig/inspect log が未実装の場合でも転写自体は成功させる
            pass

    return result