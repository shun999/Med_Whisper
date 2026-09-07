# BLSの文字起こし・自動評価

Geminiの文字起こしを入力として、18項目の意味・参加者の発言・前後関係を評価します。
通常の採点は自動で完結します。研究用の正解ラベル作成だけは音声を人が確認します。
手技の品質・圧迫の深さや速さ・医学的説明の正確性は対象ではありません。

## 採点の意味

全18項目が常に対象で、AEDを使えない条件も常に成立する想定です。
`round(達成数 / 18 * 100, 1)` で計算し、同じ項目の繰り返しは1回分です。

| 保存する状態 | 表示 | 加点 |
| --- | --- | --- |
| `met` | 達成 | あり |
| `not_detected` | 未検出：文字起こしに根拠がない | なし |
| `uncertain` | 判定不能：意味・発言者・順序などが不明 | なし |

未検出は実際の未実施を意味しません。API失敗・JSON不正・18項目の欠落は処理失敗とし、0点の評価結果を作りません。
項目の引用が原文にない場合や、参加者の発言だと確認できない場合は判定不能に変更します。
Geminiが出した状態を`model_status`に、検証後の状態を`status`に保存します。
信頼度の数値を校正済み確率のように表示することはありません。

- 「救急車を呼んでください」などの言い換えを認めます。
- 「脈なし、呼吸なし」は9・10の両方の根拠にできます。
- 11・16は圧迫回数の数唱があれば対象です。30回完遂や再開宣言は要求しません。
- 14の交代合図を圧迫の数唱として重複加点しません。11・16も別の数唱を必要とします。
- 15には後続のショック、16には先行するショック完了の根拠を要求します。
- 17は救急隊へ引き継ぐ想定で経緯を説明できれば対象で、説明内容の正確性は問いません。
- AED機器の音声・指導者の助言は加点せず、時間順序の補助根拠にだけ使用できます。

### 発言者と文字起こしの限界

初期設定は日本語・逐語モード・カスタム語彙です。Geminiの仕様上、カスタム語彙は話者分離・単語時刻と併用できません。
そのため、この評価器は声から話者を識別しているわけではありません。文脈で区別できない箇所は保守的に判定不能とします。
特にAED案内の近くの「離れてください」は、参加者が実際に発していても判定不能になることがあります。
逆に、文脈判定には誤りが残り得るため、参加者以外を完全に除外できる保証はありません。
根拠の`start`・`end`は原文の文字位置（Pythonの文字列添字、終了側を含まない）で、音声の秒数ではありません。
欠落したコールを評価器が推測で補うことはありません。

公式仕様: [文字起こし](https://ai.google.dev/gemini-api/docs/transcribe)、[構造化出力](https://ai.google.dev/gemini-api/docs/structured-output)。

## 実行

Python 3.12と既存のgoogle-genai依存関係を使用します。仮想環境ファイルを手で編集する必要はありません。

```bash
uv run jupyter lab
```

`gemini.ipynb`を開き、上から実行します。初期入力は既存TXTです。
`INPUT_MODE = "audio"`にすると対象音声を文字起こししてから評価します。
APIキーは非表示で入力できます。空欄の場合、キャッシュのある処理のみ実行できます。
バッチは`RUN_BATCH = True`にした場合だけ動作します。開発用/検証用の切り替えは`BATCH_SPLIT`です。

CLIでは`GEMINI_API_KEY`（または`GOOGLE_API_KEY`）を実行環境に設定します。
シェル履歴へキー自体を残さない入力例（`.env`等には保存しません）:

```bash
read -rsp 'Gemini API key: ' GEMINI_API_KEY
export GEMINI_API_KEY
```

```bash
# 既存TXT：採点APIだけを呼ぶ
uv run python scripts/evaluate_bls.py --transcript outputs/transcription/gemini/1回目_右前_gemini.txt

# 音声1本：文字起こし＋採点
uv run python scripts/evaluate_bls.py --audio data/0604data/1回目_右前.wav

# 実行前の確認：通信・ファイル保存なし
uv run python scripts/evaluate_bls.py --audio-dir data/20260706 --dry-run

# 全20本（固定検証の前は、ノートブックで開発用だけを選択）
uv run python scripts/evaluate_bls.py --audio-dir data/20260706

# 文字起こしだけ：評価モデルのリクエストを節約
uv run python scripts/evaluate_bls.py --audio-dir data/20260706 --transcribe-only

# 語彙のA/B比較：旧語彙の不可視文字を除去した設定と比較
uv run python scripts/evaluate_bls.py --audio-dir data/20260706 --vocabulary baseline
```

WAV/MP3/FLAC/M4A/OGGとMP4/MOVを受け付けます。動画はffmpegで一時領域にモノラル16kHz WAVを作り、送信後に一時領域を片付けます。
入力のdata/や既存のTXTには書き込みません。Gemini上の一時アップロードも処理終了時に削除を試みます。
`--audio-dir`は直下のみを対象とします。同じ内容を音声と動画の両方で指定するとそれぞれが対象になるので、対象一覧を確認してください。

終了コードは成功0、処理失敗・上限中断2です。上限で中断したときは後で同じコマンドを実行すると、成功済みキャッシュを再利用します。
バッチの失敗を0点として一覧に混ぜることはありません。

## 保存先・再現性・API上限

標準の保存先は`outputs/evaluation/bls/`です。

| 場所 | 内容 |
| --- | --- |
| `cache/transcription/` | 入力ハッシュ・語彙・モデルごとの原文TXT、設定、API生応答JSON |
| `cache/evaluation/` | 原文・モデル・評価基準・プロンプト・JSONスキーマごとの成功応答 |
| `responses/` | 採点APIの生応答。JSON解析に失敗した応答も保存 |
| `results/` | 原文、18項目、総合点、根拠、モデルの判断、検証上の留保を含むJSONとCSV |
| `runs/` | バッチの対象一覧、結果への参照、失敗、中断時の未処理一覧 |

ファイル名は内容と設定のハッシュを含みます。別の設定の実験結果は上書きしません。
同一設定の実行は最初に得た応答を再利用するため、Geminiを再度呼ぶことによる変動を避けられます。
これはモデルの完全な決定性を保証するものではありません。実際の応答・作成時刻・モデル名も記録します。
検証コードを変更した場合は`VALIDATOR_VERSION`を更新すると、新しい検証結果を別ファイルに保存できます。
語彙変更は文字起こし、評価基準変更は採点キャッシュを無効化します。同じ原文になった場合は採点キャッシュを共有します。

`--output-dir`で独立した実験ディレクトリを選べますが、書き込み先はこのプロジェクトのoutputs/以下か一時ディレクトリ以下に限定しています。
同じプロジェクトのAPI予算は出力先を変えても共有されます。`outputs/.bls_state/`はGit管理外です。

文字起こしは`gemini-3.5-transcribe`、採点は`gemini-3.8-flash`が初期値です。
それぞれのローカル上限は3回/分・25回/日で、`--transcription-rpm/rpd`、`--evaluation-rpm/rpd`で設定できます。
Google側の実際の上限がこの設定以上であることは保証しません。AI Studioでプロジェクト・モデル別に確認してください。
上限はAPIキー別ではなくプロジェクト別です。同じGoogleプロジェクトを使う複数の起動では同じ`--quota-scope`を使用してください。
別アプリや別マシンから消費した枠はローカル台帳では把握できず、Google側の429が優先されます。
日付は米国太平洋時間で切り替えます。[公式の制限仕様](https://ai.google.dev/gemini-api/docs/rate-limits)

成功・失敗を含む推論リクエストの試行回数を数え、429と一部5xxは最大3試行です。ファイルのアップロード/削除回数は推論枠には含めません。
SDK側の内部再試行も無効化しています。Pythonからクライアントを渡す場合は`create_gemini_client(api_key)`を使用してください。
モデルの利用不可・認証失敗で別モデルに自動変更したり、課金プランを変更したりはしません。

## 精度検証

初期の正解雛形は`outputs/evaluation/bls/gold_20260706.json`です。
再生成が必要な場合は新しい保存先を指定します。既存の正解ファイルは上書きしません。

```bash
uv run python scripts/benchmark_bls.py init --audio-dir data/20260706 --output outputs/evaluation/bls/gold_new.json
```

各sampleの`labels`は1〜18の`true`（達成）/`false`（未達成）/`null`（未注釈）です。
AGENTS.mdで100点と指定された2〜5回目だけを事前にtrueとし、その他はnullです。
これらの正解や100点という情報を採点プロンプトには渡しません。
音声から人が確認した根拠は`reference_evidence`、逐語録は`reference_text`、注釈者等は`label_source`に記入します。
原音声を聞かずにASR出力や採点モデルから正解を作らないでください。

開発用は1〜3・6〜12回目、固定検証用は4・5・13〜20回目です。
語彙・評価基準・プロンプトを開発用で調整し、その後に固定検証用を評価します。
テキストの言い換えテストは合成例であり、音声認識の実測精度とは区別します。

```bash
# --runにはevaluate_bls.pyまたはノートブックが表示した実際のrun JSONを指定
uv run python scripts/benchmark_bls.py report --manifest outputs/evaluation/bls/gold_20260706.json --run outputs/evaluation/bls/runs/RUN.json --output outputs/evaluation/bls/validation_report.json

# 開発用のレポート
uv run python scripts/benchmark_bls.py report --manifest outputs/evaluation/bls/gold_20260706.json --run outputs/evaluation/bls/runs/RUN.json --split development --output outputs/evaluation/bls/development_report.json

# 人が確認したreference_textの採点。別の採点リクエストが必要
uv run python scripts/benchmark_bls.py references --manifest outputs/evaluation/bls/gold_20260706.json --output outputs/evaluation/bls/reference_run.json

# ASRによる採点と、人が確認した逐語録による採点を並べて切り分け
uv run python scripts/benchmark_bls.py report --manifest outputs/evaluation/bls/gold_20260706.json --run outputs/evaluation/bls/runs/RUN.json --reference-run outputs/evaluation/bls/reference_run.json --output outputs/evaluation/bls/diagnostic_report.json
```

レポートは項目別・全体の適合率/再現率/F1、判定不能率、点数MAE、誤判定一覧、評価済み/未処理/未注釈のcoverageを含みます。
分母が0の指標はnullです。未注釈をfalseにしません。総合点MAEは18項目すべてが注釈済みの音声だけで計算します。
人の逐語録がある場合のCERはNFKC正規化と空白除去後の文字編集距離/正解文字数で、句読点は残します。
同じ音声のbaselineとrevisedのrunからそれぞれレポートを作り、CER・見逃し・誤加点を比較します。
逐語録が未入力ならCERを計算したように見せずnullにします。

人の逐語録では正しく判定しASRでは誤る場合、`stage_hint=asr_or_asr_context`を付けます。
人の逐語録でも誤る場合は`evaluator_also_failed`です。文脈とモデル変動も関係するので原因の断定ではありません。
全項目達成例だけでは誤加点を評価できないため、未達成のラベルと陰性の合成テストも必要です。
既知4本の自動100点は精度目標であり、期待点への補正や人の修正を自動精度に混ぜません。
同一収録日の20本であり、別の日・話者・環境への一般化精度を示すものではありません。

## オフライン検証

```bash
uv run python -m unittest discover -s tests -v
```

ネットワークも実データも使わず、合成例と模擬APIで採点・根拠検証・上限・キャッシュ・再開・集計を検証します。
これらのテストの成功は、実際のGeminiが正しく文字起こし・意味判定することを保証しません。
