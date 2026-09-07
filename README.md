# MedWhisper

BLS（一次救命処置）演習動画の音声を解析し、必要なコールの実施状況を評価するためのプロジェクトです。

## 主な解析

Geminiによる文字起こしと18項目の自動採点は [gemini.ipynb](gemini.ipynb) です。
保存済みTXTからの採点、音声の一括処理、根拠付きJSON/CSV出力に対応します。
採点仕様・API上限・精度検証手順は [BLS評価システム](docs/bls_evaluation.md) を参照してください。

```bash
# 通信せず対象とリクエスト数を確認
uv run python scripts/evaluate_bls.py --transcript outputs/transcription/gemini/1回目_右前_gemini.txt --dry-run
# GEMINI_API_KEYを設定済みの場合、保存済みTXTを採点
uv run python scripts/evaluate_bls.py --transcript outputs/transcription/gemini/1回目_右前_gemini.txt
# APIを使わない検証
uv run python -m unittest discover -s tests -v
```

Beam Search カスタム Whisper による解析は [263_full_paper.ipynb](263_full_paper.ipynb) に残しています。

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
│   └── transcription/      # 既存のGemini / Whisper文字起こし結果
├── 263_full_paper.ipynb   # 現行の解析ノートブック
└── requirements.txt
```

## データ配置

解析対象は `data/0604data/` または `data/0606data/` に配置します。動画・音声や仮想環境は GitHub にコミットしないでください。

## 実行例

Python 3.12 と依存関係は `uv` で管理・実行します。既存の `env/` は使用せず、プロジェクトルートで同期してください。

```bash
uv python install 3.12
uv sync
uv run python --version
uv run python scripts/audio_explorer.py data/0604data/example.wav --no-diar
uv run python scripts/check_model_loading.py
```

ノートブックは同じ環境から起動します。

```bash
uv run jupyter lab
```

VS Codeでは `.venv/bin/python` をPythonインタープリターとして選択してください。

GPU認識は次のコマンドで確認できます。

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

本プロジェクトはCUDA 12.1版PyTorchを使用します。`False`になりドライバー互換性の警告が出る場合、コードはCPUへフォールバックしますが、GPUを使用するにはNVIDIAドライバーの更新が必要です。

`notebooks/archive/` の過去実験も実行する場合は、追加依存関係を同期します。

```bash
uv sync --group archive
uv run jupyter lab
```

各スクリプトとノートブックはプロジェクトのルートディレクトリから実行してください。生成物は用途ごとに `outputs/` 配下へ保存されます。`outputs/` は実験結果を共有するため、Git 管理対象として維持します。
