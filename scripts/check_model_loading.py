"""Medical Whisper モデルを手動で読み込むスモークチェック。"""

import os

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def main() -> None:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN 環境変数を設定してください")

    repo = "Na0s/Medical-Whisper-Large-v3"
    AutoProcessor.from_pretrained(repo, token=hf_token, trust_remote_code=True)
    AutoModelForSpeechSeq2Seq.from_pretrained(
        repo,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype="auto",
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()


if __name__ == "__main__":
    main()
