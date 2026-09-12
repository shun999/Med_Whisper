# MedWhisper

BLS（一次救命処置）演習動画の音声を解析し、必要なコールの実施状況を評価するためのプロジェクトです。

## Geminiによる評価

Geminiによる文字起こしと18項目の自動採点は [gemini.ipynb](gemini.ipynb) またはCLIから実行できます。保存済みTXTからの採点、音声・動画の一括処理、根拠付きJSON/CSV出力に対応します。採点仕様、API上限、精度検証手順は [BLS評価システム](docs/bls_evaluation.md) を参照してください。

採点前に`data/LED音声人間文字起こし.txt`を参照し、一致する機器アナウンスを人間の発言候補から分離します。分離した機器音声は直接加点せず、ショック完了などの時点確認にだけ使用します。参照TXTはローカルに配置し、別の場所にある場合は`--device-reference PATH`で指定してください。

採点実行後には、音声別の点数と`data/人間音声採点データ.csv`の模範点数との差、平均絶対誤差をまとめたTXTを`outputs/evaluation/bls/summaries/`に自動保存します。模範CSVは比較にだけ使用し、採点モデルには渡しません。

```bash
# 通信せず対象とリクエスト数を確認
uv run python scripts/evaluate_bls.py --transcript outputs/transcription/gemini/1回目_右前_gemini.txt --dry-run

# GEMINI_API_KEYを設定済みの場合、保存済みTXTを採点
uv run python scripts/evaluate_bls.py --transcript outputs/transcription/gemini/1回目_右前_gemini.txt

# 音声または動画を文字起こしして採点
uv run python scripts/evaluate_bls.py --audio data/0604data/example.wav

# APIを使わない検証
uv run python -m unittest discover -s tests -v
```

## ディレクトリ構成

```text
.
├── src/                       # Gemini連携・評価・精度集計
├── scripts/                   # 評価とベンチマークのCLI
├── tests/                     # API通信を行わない自動テスト
├── docs/                      # 採点仕様・検証手順
├── data/                      # 入力動画・音声（Git管理対象外）
├── outputs/
│   ├── evaluation/bls/        # キャッシュ・採点結果・実行履歴・正解ラベル
│   └── transcription/gemini/  # 保存済みのGemini文字起こし
├── gemini.ipynb               # 対話的な評価ノートブック
├── pyproject.toml             # uv用の依存関係定義
└── uv.lock                    # 固定済み依存関係
```

## セットアップ

Python 3.12と依存関係はuvで管理します。既存の`env/`は使用せず、プロジェクトルートで同期してください。

```bash
uv python install 3.12
uv sync
uv run python --version
uv run jupyter lab
```

MP4/MOV入力から音声を一時変換するため、システムの`ffmpeg`コマンドが必要です。

```bash
ffmpeg -version
```

CLIでGemini APIを使用する場合は、`GEMINI_API_KEY`または`GOOGLE_API_KEY`を実行環境に設定します。APIキーや`.env`などの秘密情報はGitHubにコミットしないでください。

## データと出力

解析対象は`data/`配下に配置します。精度評価には`data/20260706/`のWAVファイルを使用します。`data/`内の動画・音声は編集せず、GitHubにもコミットしないでください。

各スクリプトとノートブックはプロジェクトルートから実行してください。Geminiのキャッシュ、採点結果、実行履歴はAPI無料枠の節約と再現性確保のため`outputs/evaluation/bls/`に保持します。
