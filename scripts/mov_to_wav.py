import ffmpeg
import sys
import os
from pathlib import Path

def convert_mov_to_wav(input_file, output_file):
    """
    MOVファイルをWAVファイルに変換します。

    :param input_file: 入力MOVファイルのパス
    :param output_file: 出力WAVファイルのパス
    """
    if not os.path.exists(input_file):
        print(f"エラー: 入力ファイルが見つかりません: {input_file}")
        return

    print(f"変換を開始します: {input_file} -> {output_file}")
    try:
        # ffmpegのプロセスを定義
        # -i: 入力ファイル
        # -acodec pcm_s16le: オーディオコーデックをWAV標準のPCM 16bitに指定
        # -ar 44100: サンプリングレートを44.1kHzに指定（CD音質）
        stream = ffmpeg.input(input_file)
        stream = ffmpeg.output(stream.audio, output_file, acodec='pcm_s16le', ar=44100)
        
        # 既存の出力ファイルを上書きする設定で実行
        ffmpeg.run(stream, overwrite_output=True)
        
        print(f"変換が完了しました: {output_file}")
    except ffmpeg.Error as e:
        print("エラーが発生しました:")
        print(e.stderr.decode())

if __name__ == '__main__':
    # --- ここを編集してください ---
    project_root = Path(__file__).resolve().parents[1]
    input_mov_file = project_root / "data" / "20241018" / "右後_2回目_川村先生.MOV"
    output_wav_file = input_mov_file.with_suffix(".wav")

    # スクリプトと同じディレクトリにあると仮定
    # 必要に応じて絶対パスを指定してください (例: "C:/Users/YourUser/Videos/test.mov")
    
    convert_mov_to_wav(input_mov_file, output_wav_file)
