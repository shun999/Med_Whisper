from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import os, torch

HF_TOKEN = os.getenv("HF_TOKEN") or "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"  # ←必ず有効なReadトークン

repo = "Na0s/Medical-Whisper-Large-v3"

processor = AutoProcessor.from_pretrained(
    repo, token=HF_TOKEN, trust_remote_code=True
)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    repo, token=HF_TOKEN, trust_remote_code=True, torch_dtype="auto"
).to("cuda" if torch.cuda.is_available() else "cpu").eval()
