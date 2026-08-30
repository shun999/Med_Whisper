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

## BLSコールの採点

文字起こし済みのUTF-8テキストは、音声処理と独立して採点できます。規則ベースのみを試す場合はAPIキー不要です。

```bash
uv run python scripts/evaluate_bls_calls.py \
  outputs/transcription/gemini/1回目_右前_gemini5.txt \
  --method rules
```

規則ベースとGemini判定を比較する場合は、`GEMINI_API_KEY`を環境変数に設定するか、対話端末で表示される非表示入力欄へ入力します。キーをファイルやコマンドライン引数へ保存しないでください。

```bash
uv run python scripts/evaluate_bls_calls.py transcript.txt \
  --method both \
  --llm-model gemini-3.7-flash \
  --llm-repeats 3 \
  --sample-id session01-gemini \
  --session-id session01
```

12項目を`0 / 0.5 / 1`で等配点し、内容スコアを0〜100で出力します。順序スコアは内容スコアへ合算せず、比較できた制約数の`sequence_coverage`と一緒に表示します。判定結果は`outputs/evaluation/`のJSONとCSVへ保存され、根拠文、文字位置、判定理由、LLMの3回分の結果を確認できます。ルーブリックは`configs/bls_call_rubric.json`で版管理しています。

### 研究用の正解データと精度評価

同一演習の別カメラ・別マイクは同じ`session_id`にし、セッション単位で開発用70%と固定テスト用30%へ分けます。次の列を持つmanifest CSVを用意します。

```csv
session_id,sample_id,transcript_path,audio_path,split
session01,session01-gemini,outputs/transcription/gemini/session01.txt,data/session01.wav,dev
session02,session02-gemini,outputs/transcription/gemini/session02.txt,data/session02.wav,test
```

専門家・非専門家が独立評価する注釈シートを生成します。テキスト注釈は採点器自体の評価、原音注釈はASRを含むEnd-to-End評価に使います。`score`にはコール項目で`0 / 0.5 / 1`、順序項目で`0 / 1`を入力し、不一致を確定した行を追加する場合は`annotator_role=adjudicated`とします。

```bash
uv run python scripts/validate_bls_scoring.py template \
  manifest.csv outputs/evaluation/annotations.csv
```

採点JSONと固定テスト注釈から、項目一致率、macro-F1、重み付きCohenのκ、内容スコアMAE、順序精度、セッション単位のブートストラップ95%信頼区間を計算します。

```bash
uv run python scripts/validate_bls_scoring.py metrics \
  outputs/evaluation/annotations.csv \
  outputs/evaluation/*_evaluation.json \
  --split test \
  --output outputs/evaluation/validation_report.json
```

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
