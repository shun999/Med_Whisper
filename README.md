# MedWhisper

BLS（一次救命処置）演習動画の音声を解析し、必要なコールの実施状況を評価するためのプロジェクトです。

## 主な解析

最新の Beam Search カスタム Whisper による解析は [263_full_paper.ipynb](263_full_paper.ipynb) です。ノートブックは既存の相対パスとの互換性を保つため、当面はプロジェクト直下に配置しています。

## ディレクトリ構成

```text
.
├── src/                    # Whisper 拡張などの再利用コード
├── scripts/                # 音声変換・話者分離・解析用スクリプト
├── notebooks/archive/      # 過去の実験ノートブック
├── data/                   # 入力動画・音声（Git 管理対象外）
├── _work/                  # 再利用する既存実験の中間データ
├── outputs/
│   ├── audio_explorer/     # 音響特徴の解析結果
│   ├── beam_search/        # Beam Search のログ・CSV・図
│   ├── diarization/        # pyannote の話者分離結果
│   ├── evaluation/         # 候補文・評価用 CSV
│   └── transcription/      # Whisper の文字起こし結果
├── 263_full_paper.ipynb   # 現行の解析ノートブック
└── requirements.txt
```

## データ配置

解析対象は `data/0604data/` または `data/0606data/` に配置します。動画・音声や仮想環境は GitHub にコミットしないでください。

## 実行例

依存関係は `uv` で管理・実行します。

```bash
uv sync
uv run python scripts/audio_explorer.py data/0604data/example.wav --no-diar
uv run python scripts/check_model_loading.py
```

各スクリプトとノートブックはプロジェクトのルートディレクトリから実行してください。生成物は用途ごとに `outputs/` 配下へ保存されます。`outputs/` は実験結果を共有するため、Git 管理対象として維持します。
