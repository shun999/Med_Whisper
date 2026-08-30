import argparse
from pathlib import Path

import ffmpeg

def convert_mov_to_wav(input_file, output_file):
    """
    MOVファイルをWAVファイルに変換します。

    :param input_file: 入力MOVファイルのパス
    :param output_file: 出力WAVファイルのパス
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    if not input_file.is_file():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"変換を開始します: {input_file} -> {output_file}")
    try:
        # ffmpegのプロセスを定義
        # -i: 入力ファイル
        # -acodec pcm_s16le: オーディオコーデックをWAV標準のPCM 16bitに指定
        # -ar 44100: サンプリングレートを44.1kHzに指定（CD音質）
        stream = ffmpeg.input(str(input_file))
        stream = ffmpeg.output(stream.audio, str(output_file), acodec='pcm_s16le', ar=44100)
        
        # 既存の出力ファイルを上書きする設定で実行
        ffmpeg.run(stream, overwrite_output=True)
        
        print(f"変換が完了しました: {output_file}")
    except ffmpeg.Error as exc:
        message = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
        raise RuntimeError(f"ffmpegによる変換に失敗しました:\n{message}") from exc

def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="MOV動画の音声をWAVへ変換します")
    parser.add_argument("input", type=Path, help="入力MOVファイル")
    parser.add_argument("--output", type=Path, default=None, help="出力WAVファイル")
    args = parser.parse_args()

    input_file = args.input.resolve()
    output_file = args.output or (
        project_root / "outputs" / "audio" / f"{input_file.stem}.wav"
    )
    convert_mov_to_wav(input_file, output_file)


if __name__ == '__main__':
    main()
