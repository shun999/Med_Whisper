from pydub import AudioSegment
from pathlib import Path

# m4a があるフォルダ
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "data" / "2025115cleandata" / "スマホ"  # ←ここだけ変える

# 「ファイル名: [(開始秒, 終了秒), ...]」
# 終了を「最後まで」にしたいところは None にする
split_plan = {
    "3_川村先生_協力者_大野.m4a": [
        (0, 19),
        (20, 35),
        (35, 50),
        (50, 72),
        (72, 82),
        (82,109),
        (109, 125),
        (125, None),   # 最後まで
    ],
    "4_川村先生_協力者_大場AED.m4a": [
        (0, 17),
        (17, 33),
        (33, 47),
        (47, 67),
        (67, 79),
        (90, 98),
        (102, 121),
        (121, None),
    ],
    "5_川村先生_シンプル1.m4a": [
        (0, 19),
        (19, 40),
        (40, 56),
        (56, 77),
        (77, 88),
        (88, 111),
        (111, 126),
        (126, None),
    ],
}

OUT_DIR = PROJECT_ROOT / "outputs" / "segments"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, segments in split_plan.items():
        in_path = BASE_DIR / fname
        if not in_path.is_file():
            print(f"[WARN] 入力がないためスキップ: {in_path}")
            continue
        print(f"[INFO] Processing {in_path}")

        # m4a を読み込み（自動で ffmpeg が使われる）
        audio = AudioSegment.from_file(in_path)

        for idx, (start_s, end_s) in enumerate(segments, start=1):
            start_ms = int(start_s * 1000)
            end_ms = len(audio) if end_s is None else int(end_s * 1000)
            seg = audio[start_ms:end_ms].set_frame_rate(16000).set_channels(1)
            out_path = OUT_DIR / f"{in_path.stem}_part{idx:02d}.wav"
            seg.export(out_path, format="wav")
            end_label = end_s if end_s is not None else "end"
            print(f"  -> saved {out_path} ({start_s}–{end_label} s)")


if __name__ == "__main__":
    main()
