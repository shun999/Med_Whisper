#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyannote を「生波形」と「音声加工後」の両方に適用し、
話者分離結果（区間CSV、話者別WAV、連結WAV）と
時系列グラフ（RMS×話者）を保存するワンファイルスクリプト。

出力先:
  pyannote_result/
    timeline_raw.png
    timeline_processed.png
    raw/
      spk_SPEAKER_00/*.wav, ...
      _concat/spk_SPEAKER_00_concatenated.wav, ...
      segments_raw.csv
    processed/
      spk_SPEAKER_00/*.wav, ...
      _concat/spk_SPEAKER_00_concatenated.wav, ...
      segments_processed.csv
"""

# ========= ブロック0: 既存のtorchaudioパッチ + pyannoteパイプライン =========
import sys, types, torchaudio

def _noop(*a, **k): pass
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = _noop
if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
torchaudio.set_audio_backend("soundfile")

mod_backend = types.ModuleType("torchaudio.backend")
mod_common  = types.ModuleType("torchaudio.backend.common")
try:
    from torchaudio import AudioMetaData as _AMD
except Exception:
    class _AMD: pass
mod_common.AudioMetaData = _AMD
sys.modules["torchaudio.backend"] = mod_backend
sys.modules["torchaudio.backend.common"] = mod_common

import soundfile as sf
if not hasattr(torchaudio, "info"):
    def _info(path):
        try:
            f = sf.SoundFile(path)
            sr = f.samplerate
            frames = len(f)
            channels = f.channels
            f.close()
            class Info:
                sample_rate = sr
                num_channels = channels
                num_frames = frames
            return Info()
        except Exception as e:
            raise RuntimeError(f"torchaudio.info fallback failed for {path}: {e}")
    torchaudio.info = _info

from pyannote.audio import Pipeline
import os

HF_TOKEN = os.getenv("HF_TOKEN", "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")  # ← 環境変数で上書き推奨
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
# pipe = pipe.to("cuda")  # GPUあれば有効化

print("pyannote pipeline ready ✓")

# ========================== 以降: 本処理 ===============================
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from pathlib import Path
from collections import defaultdict
import csv

# ---- 設定 ----
FILE_NAME = "20240827/右後_4回目_大野"
BASE_DIR = Path("/root/MedWhisper")
WAV_RAW  = BASE_DIR / f"{FILE_NAME}.wav"   # 生波形
OUTROOT  = Path("./pyannote_result")       # 要求どおり: 直下にグラフ。配下に raw/ processed/
OUTROOT.mkdir(parents=True, exist_ok=True)

# ---- ユーティリティ ----
def rms_db(x):
    return 20*np.log10(np.sqrt(np.mean(x**2))+1e-12)
def peaking_eq(x, fs, f0=1500.0, Q=1.0, gain_db=3.0):
    import numpy as np
    import scipy.signal as sig

    A = 10**(gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * Q)
    cw = np.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cw
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cw
    a2 = 1 - alpha / A

    # 正規化
    b0 /= a0; b1 /= a0; b2 /= a0
    a1 /= a0; a2 /= a0

    # SOS 形: [b0, b1, b2, a0(=1), a1, a2]
    sos = np.array([[b0, b1, b2, 1.0, a1, a2]], dtype=np.float64)
    y = sig.sosfiltfilt(sos, x).astype(np.float32)
    return y


def bandshape_for_asr(y, sr, hpf=80, lpf=8000, peak_f0=1500.0, peak_q=1.0, peak_db=3.0):
    # 安定なバターワース(HPF/LPF)
    nyq = sr/2
    lpf_safe = min(lpf, 0.95*nyq)
    sos = butter(4, hpf, btype="highpass", fs=sr, output="sos"); y2 = sosfiltfilt(sos, y)
    sos = butter(4, lpf_safe, btype="lowpass",  fs=sr, output="sos"); y2 = sosfiltfilt(sos, y2)
    # 簡易ピーキングEQ (+3dB)
    # （sos化していない簡易版。極端な値は入れない想定）
    y2 = peaking_eq(y2, fs=sr, f0=peak_f0, Q=peak_q, gain_db=peak_db)
    return np.clip(y2, -1.0, 1.0)

def merge_close_segments(seg_list, max_gap=0.4, min_keep=0.6):
    # seg_list: [(spk, start, end)]
    seg_list = sorted(seg_list, key=lambda x: (x[0], x[1], x[2]))
    from itertools import groupby
    merged = []
    for spk, group in groupby(seg_list, key=lambda x: x[0]):
        group = list(group)
        cur_s, cur_e = group[0][1], group[0][2]
        for _, s, e in group[1:]:
            if s - cur_e <= max_gap:
                cur_e = max(cur_e, e)
            else:
                if (cur_e - cur_s) >= min_keep:
                    merged.append((spk, cur_s, cur_e))
                cur_s, cur_e = s, e
        if (cur_e - cur_s) >= min_keep:
            merged.append((spk, cur_s, cur_e))
    merged.sort(key=lambda x: x[1])
    return merged

def pad_segments(seg_list, pad=0.2, total_dur=None):
    out = []
    for spk, s, e in seg_list:
        s2 = max(0.0, s - pad)
        e2 = e + pad if total_dur is None else min(total_dur, e + pad)
        out.append((spk, s2, e2))
    return out

def write_segments_csv(csv_path, segments):
    # segments: [(spk, start, end)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "start_sec", "end_sec", "duration_sec"])
        for spk, s, e in segments:
            w.writerow([spk, f"{s:.3f}", f"{e:.3f}", f"{(e-s):.3f}"])

def plot_timeline_rms(wav_path, segments_padded, out_png):
    y, sr = sf.read(wav_path, always_2d=False)
    if y.ndim > 1: y = y.mean(axis=1)

    hop = int(0.01 * sr)
    win = int(0.032 * sr)
    num = max(1, 1 + (len(y) - win) // hop)
    pad = max(0, win + (num - 1) * hop - len(y))
    if pad > 0:
        y = np.concatenate([y, np.zeros(pad)], axis=0)

    rms = np.array([np.sqrt(np.mean(y[i*hop:i*hop+win]**2)) for i in range(num)])
    times = (np.arange(num) * hop + win/2) / sr

    buckets = defaultdict(list)
    for spk, s, e in segments_padded:
        buckets[spk].append((s, e))

    plt.figure(figsize=(12, 4))
    for spk, spans in sorted(buckets.items()):
        arr = np.zeros_like(rms)
        for s, e in spans:
            mask = (times >= s) & (times < e)
            arr[mask] = rms[mask]
        plt.plot(times, arr, label=spk, linewidth=1.0)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (RMS)")
    plt.title(Path(wav_path).name + " | Speaker-wise RMS over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def diarize_and_export(label, wav_path, outroot: Path):
    """
    label: "raw" or "processed"
    wav_path: 入力wav（モノ or ステレオ可）
    出力:
      outroot/{label}/spk_*/*.wav
      outroot/{label}/_concat/spk_*_concatenated.wav
      outroot/segments_{label}.csv（区間）
      outroot/timeline_{label}.png（グラフは直下）
    """
    out_dir = outroot / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) diarization
    dia = pipe(wav_path, min_speakers=2, max_speakers=3)
    segments = []
    for turn, _, speaker in dia.itertracks(yield_label=True):
        segments.append((speaker, float(turn.start), float(turn.end)))

    # 2) 後処理
    total_dur = librosa.get_duration(path=wav_path)
    segments_merged = merge_close_segments(segments, max_gap=0.4, min_keep=0.6)
    segments_padded = pad_segments(segments_merged, pad=0.2, total_dur=total_dur)

    # 3) 書き出し: 区間CSV
    write_segments_csv(outroot / f"segments_{label}.csv", segments_padded)

    # 4) 話者別に切り出し保存
    y_base, sr_base = librosa.load(wav_path, sr=None, mono=True)
    buckets = defaultdict(list)
    for spk, s, e in segments_padded:
        buckets[spk].append((s, e))

    for spk, spans in buckets.items():
        spkdir = out_dir / f"spk_{spk}"
        spkdir.mkdir(exist_ok=True)
        count = 0
        for i, (s, e) in enumerate(spans):
            s0, s1 = int(s*sr_base), int(e*sr_base)
            if s1 > s0:
                seg = y_base[s0:s1]
                sf.write((spkdir / f"{i:02d}_{spk}_{s:.2f}-{e:.2f}.wav").as_posix(), seg, sr_base)
                count += 1
        print(f"[{label}] {spk}: saved {count} segments in {spkdir}")

    # 5) 連結
    concat_dir = out_dir / "_concat"
    concat_dir.mkdir(exist_ok=True)
    speaker_dirs = sorted([d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith('spk_')])
    for spk_dir in speaker_dirs:
        speaker_name = spk_dir.name
        wav_files = sorted(spk_dir.glob("*.wav"))
        if not wav_files:
            print(f"[{label}] {speaker_name}: no wavs, skip")
            continue
        segs = []
        for p in wav_files:
            y_seg, sr_seg = sf.read(p, always_2d=False)
            if y_seg.ndim > 1: y_seg = y_seg.mean(axis=1)
            segs.append(y_seg.astype(np.float32))
        concatenated = np.concatenate(segs) if segs else np.array([], dtype=np.float32)
        out = concat_dir / f"{speaker_name}_concatenated.wav"
        sf.write(out.as_posix(), concatenated, sr_base)
        print(f"[{label}] concat -> {out.name} ({len(concatenated)/sr_base:.2f}s)")

    # 6) グラフ保存（直下）
    plot_timeline_rms(wav_path, segments_padded, outroot / f"timeline_{label}.png")

def main():
    # a) 生波形の読み込み
    y_raw, sr_raw = librosa.load(WAV_RAW, sr=None, mono=True)

    # b) 音声加工（ASR想定の軽整形）
    y_proc = bandshape_for_asr(y_raw, sr_raw, hpf=80, lpf=8000, peak_f0=1500.0, peak_q=1.0, peak_db=3.0)
    wav_processed = OUTROOT / "processed_input.wav"
    sf.write(wav_processed.as_posix(), y_proc, sr_raw)
    print(f"[processed] wrote {wav_processed.name}  peak={np.max(np.abs(y_proc)):.3f}  RMS(dBFS)={rms_db(y_proc):.1f}")

    # c) 生波形で diarization → 保存
    diarize_and_export("raw", WAV_RAW.as_posix(), OUTROOT)

    # d) 加工後で diarization → 保存
    diarize_and_export("processed", wav_processed.as_posix(), OUTROOT)

    print("\nAll done. Results ->", OUTROOT.resolve())

if __name__ == "__main__":
    main()
