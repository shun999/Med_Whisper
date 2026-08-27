# Repository Guidelines

## プロジェクトの概要
このプロジェクトの目的は、一次救命処置（BLS）の演習の様子を撮影した動画ファイルの音声解析をし、一次救命処置（BLS）に必要な発言（コール）がどれだけ行われたかを定量的に評価することです。
可能であれば、動画ファイルの音声解析をし、0から100までのスコアを算出し、一次救命処置（BLS）の演習の定量的な評価を自動的に行うことを目指します。

## 開発環境の詳細
Operating System: Ubuntu 20.04.6 LTS  
Kernel: Linux 5.4.0-153-generic  
Architecture: x86-64  
GPU: NVIDIA RTX A6000  

## Project Context
- バージョン管理はuvで行う  
- コード管理はGitHubで行う

## 注意事項
- 仮想環境ファイル(envなど)は編集せず、GitHubにアップロードしないこと
- /dataディレクトリには、生の動画データがあるため、絶対編集しないこと
- /dataディレクトリはGitHubにアップロードしないこと

