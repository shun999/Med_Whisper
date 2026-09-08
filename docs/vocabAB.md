• この記述は、文字起こし用カスタム語彙の効果をA/B比較するためのものです。

  uv run python scripts/evaluate_bls.py \
    --audio-dir data/20260706 \
    --vocabulary baseline

  各指定の意味は次のとおりです。

  - uv run：uv管理下のPython環境で実行
  - scripts/evaluate_bls.py：文字起こしとBLS評価を実行
  - --audio-dir data/20260706：ディレクトリ直下の音声・動画を一括処理
  - --vocabulary baseline：文字起こしに旧カスタム語彙を使用

  ### baselineとrevisedの違い

  通常実行時のデフォルトはrevisedです。

  uv run python scripts/evaluate_bls.py --audio-dir data/20260706

  これは以下と同じです。

  uv run python scripts/evaluate_bls.py \
    --audio-dir data/20260706 \
    --vocabulary revised

  語彙はsrc/bls_evaluation.py:34で定義されています。

   設定        内容
  ━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   baseline    従来使用していた13語程度の旧語彙
  ──────────  ──────────────────────────────────────────────────────
   revised     18評価項目を広くカバーするよう拡張した28語程度の語彙

  たとえばrevisedには、旧語彙になかった次のような語が追加されています。

  - 胸骨圧迫
  - 脈なし、呼吸なし
  - AEDを使えますか
  - 胸骨圧迫を交代
  - 離れてください
  - 電気ショック
  - 胸骨圧迫再開
  - 救急隊
  - 経緯、引き継ぎ
  - 荷物、かばん

  ### 「不可視文字を除去した旧語彙」とは

  baselineは、問題のあった不可視文字をそのまま再現する設定ではありません。

  語彙をAPIへ渡す前に、clean_vocabulary() (src/bls_evaluation.py:52)が次の処理を行います。

  - UnicodeをNFKC正規化
  - UnicodeカテゴリCfの不可視な書式文字を削除
  - 前後の空白を削除
  - 重複語を削除

  つまり、比較したいものは次の違いです。

  baseline：不可視文字だけを除去した旧語彙
  revised ：BLS評価項目に合わせて拡張・整理した新語彙

  不可視文字によるAPIへの悪影響を除外し、語彙内容そのものを変更した効果を比較できるようにしています。

  ### A/B比較の進め方

  同じ音声に対して両方を実行します。

  # A：新語彙
  uv run python scripts/evaluate_bls.py \
    --audio-dir data/20260706 \
    --vocabulary revised

  # B：旧語彙
  uv run python scripts/evaluate_bls.py \
    --audio-dir data/20260706 \
    --vocabulary baseline

  それぞれ別の文字起こしキャッシュとrun JSONが作成されます。語彙設定がキャッシュキーに含まれるため、片方の文字起こしが誤ってもう片方に流用されること
  はありません。

  比較対象は主に次の3点です。

  - CER：人が確認した逐語録に対する文字誤り率
  uv run python scripts/evaluate_bls.py \
    --audio-dir data/20260706 \
    --vocabulary baseline \
    --transcribe-only