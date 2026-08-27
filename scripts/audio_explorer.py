# audio_explorer.py
# pip install librosa soundfile pyloudnorm noisereduce webrtcvad matplotlib numpy scipy pandas
# pip install -U pyannote.audio torch==2.1.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
# 使い方例:
#   export HUGGINGFACE_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxx
#   uv run python scripts/audio_explorer.py "data/0604data/2回目_右後ろ.wav" --noise-hint 3,7

import os, sys, argparse, struct
from pathlib import Path
import numpy as np
import pandas as pd
import librosa, soundfile as sf
import matplotlib.pyplot as plt
import webrtcvad
import pyloudnorm as pyln
from scipy.signal import butter, filtfilt
from pyannote.audio import Pipeline

def rms_dbfs(y):
    return 20*np.log10(np.sqrt(np.mean(y**2))+1e-12)

def frame_audio(y, sr, frame_ms=25, hop_ms=10):
    f = int(sr*frame_ms/1000); h = int(sr*hop_ms/1000)
    n = max(0, 1+(len(y)-f)//h)
    idx = np.arange(0, n*h, h)[:,None] + np.arange(f)[None,:]
    return (y[idx], f, h)

def spectral_features(y_f, sr):
    # 1フレーム毎の簡易特徴
    S = np.abs(np.fft.rfft(y_f, axis=1))
    S = S + 1e-12
    # 平坦度
    flat = np.exp(np.mean(np.log(S), axis=1)) / np.mean(S, axis=1)
    # 中心（粗い）
    freqs = np.fft.rfftfreq(y_f.shape[1], 1/sr)
    centroid = (S*freqs).sum(axis=1)/S.sum(axis=1)
    # フラックス
    flux = np.zeros(len(S))
    flux[1:] = np.sqrt(((S[1:]-S[:-1])**2).sum(axis=1))/S.shape[1]
    # ZCR
    zcr = ((y_f[:,1:]*y_f[:,:-1])<0).mean(axis=1)
    return flat, centroid, flux, zcr

def bytes_from_frame(x):
    x16 = np.int16(np.clip(x*32768, -32768, 32767))
    return struct.pack("<%dh" % len(x16), *x16)

def vad_track(y, sr, frame_ms=30):
    vad = webrtcvad.Vad(2)  # 0..3（3が最も厳しい）
    f = int(sr*frame_ms/1000)
    n = len(y)//f
    flags = []
    for i in range(n):
        fr = y[i*f:(i+1)*f]
        flags.append(vad.is_speech(bytes_from_frame(fr), sample_rate=sr))
    return np.array(flags, dtype=bool), f

def lufs_sr(y, sr):
    meter = pyln.Meter(sr)
    return meter.integrated_loudness(y)

def estimate_snr(y, sr, noise_hint=None):
    # noise_hint = (3,7) のような秒指定 or None
    if noise_hint and len(y) >= int(noise_hint[1]*sr):
        n = y[int(noise_hint[0]*sr):int(noise_hint[1]*sr)]
    else:
        # 最静1秒
        win = sr
        if len(y) <= win:
            n = y
        else:
            powers = [np.mean(y[i:i+win]**2) for i in range(0, len(y)-win, win)]
            i_min = int(np.argmin(powers)) if powers else 0
            n = y[i_min*win:(i_min+1)*win]
    sig_p = np.mean(y**2); noise_p = np.mean(n**2)+1e-12
    return 10*np.log10((sig_p+1e-12)/noise_p)

def diarize(y, sr, wav_path, token):
    # pyannote はファイルパス入力が基本。いったん WAV に落とす（テンポラリでもOK）
    tmp = wav_path if os.path.exists(wav_path) else "_tmp_input.wav"
    if tmp == "_tmp_input.wav":
        sf.write(tmp, y, sr)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
    diar = pipeline(tmp)
    segs = []
    for turn, _, spk in diar.itertracks(yield_label=True):
        segs.append((turn.start, turn.end, spk))
    segs.sort()
    return segs

def merge_tracks(speech_flags, step, sr, spk_segs, total_len):
    # speech_flags: bool per VAD frame（frame_ms=30ms）
    # spk_segs: [(start,end,spk),...]（秒）
    # 出力：区間ラベル列（start,end,label）
    out = []
    # VADを基礎レーンに敷く
    ptr = 0
    def append_seg(s,e,label):
        if len(out)>0 and out[-1][2]==label and abs(out[-1][1]-s)<1e-6:
            out[-1] = (out[-1][0], e, label)
        else:
            out.append((s,e,label))
    # まず noise/speech を刻む
    for i,flag in enumerate(speech_flags):
        s = i*step/sr; e = min((i+1)*step/sr, total_len)
        append_seg(s, e, "SPEECH" if flag else "NOISE")

    # 次に SPEECH を話者IDに置換
    # 時間重なり最大の話者を割当（簡易）
    for i,(s,e,label) in enumerate(out):
        if label != "SPEECH": continue
        # この区間に重なる話者を探す
        overlaps = []
        for (ss,ee,spk) in spk_segs:
            ov = max(0, min(e, ee) - max(s, ss))
            if ov > 0:
                overlaps.append((ov, spk))
        if overlaps:
            spk = sorted(overlaps, reverse=True)[0][1]
            out[i] = (s,e,spk)
    return out

def summarize(segments):
    # segments: [(s,e,label)]
    dur = {}
    for s,e,l in segments:
        dur[l] = dur.get(l,0.0) + (e-s)
    tot = sum(dur.values())+1e-12
    rows = [{"label":k, "sec":round(v,2), "ratio":round(100*v/tot,1)} for k,v in sorted(dur.items(), key=lambda x:-x[1])]
    return pd.DataFrame(rows)

def plot_timeline(segments, outpng, title="Timeline"):
    plt.figure(figsize=(14,2))
    labels = list(dict.fromkeys([l for _,_,l in segments]))
    cmap = {lab: f"C{i%10}" for i,lab in enumerate(labels)}
    for s,e,l in segments:
        plt.barh(0, e-s, left=s, height=0.4, label=l, color=cmap[l])
    plt.yticks([]); plt.xlabel("Time [s]"); plt.title(title)
    handles, labs = plt.gca().get_legend_handles_labels()
    bylab = dict(zip(labs, handles))
    plt.legend(bylab.values(), bylab.keys(), bbox_to_anchor=(1.02,1), loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(outpng, dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--noise-hint", type=str, default=None, help="e.g., 3,7 (seconds)")
    ap.add_argument("--no-diar", action="store_true", help="skip diarization")
    args = ap.parse_args()

    y, sr = librosa.load(args.wav, sr=16000, mono=True)
    total_len = len(y)/sr

    # 健全性
    lufs = lufs_sr(y, sr); rms = rms_dbfs(y); snr = estimate_snr(y, sr,
        tuple(map(float, args.noise_hint.split(","))) if args.noise_hint else None)

    # フィーチャ & VAD
    y_f, f, h = frame_audio(y, sr, frame_ms=25, hop_ms=10)  # 25ms/10ms
    flat, cent, flux, zcr = spectral_features(y_f, sr)
    speech_flags, step = vad_track(y, sr, frame_ms=30)

    # ディアライゼーション
    token = os.getenv("HUGGINGFACE_TOKEN", "")
    spk_segs = []
    if not args.no_diar and token:
        spk_segs = diarize(y, sr, args.wav, token)

    # マージ
    segments = merge_tracks(speech_flags, step, sr, spk_segs, total_len)

    # 出力物
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "audio_explorer"
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_png = output_dir / "timeline.png"
    plot_timeline(segments, timeline_png, title=os.path.basename(args.wav))

    # サマリー
    df = summarize(segments)
    df.to_csv(output_dir / "summary.csv", index=False)

    # 健全性 + 代表特徴CSV
    meta = pd.Series({
        "sr": sr, "duration_sec": round(total_len,2),
        "LUFS": round(lufs,2), "RMS_dBFS": round(rms,2), "SNR_est_dB": round(snr,2)
    })
    meta.to_csv(output_dir / "health.csv")

    # 代表フレーム特徴（先頭1万点まで）
    nrec = int(min(len(flat), 10000))
    feat = pd.DataFrame({
        "frame_idx": np.arange(nrec),
        "flatness": flat[:nrec],
        "centroid": cent[:nrec],
        "flux": flux[:nrec],
        "zcr": zcr[:nrec]
    })
    feat.to_csv(output_dir / "frame_features.csv", index=False)

    # ロジックツリー風の要約（標準出力）
    print("\n=== Health ===")
    print(meta)
    print("\n=== Class summary (seconds / %) ===")
    print(df)
    print(f"\nSaved: timeline.png, summary.csv, health.csv, frame_features.csv in {output_dir}")

if __name__ == "__main__":
    main()
