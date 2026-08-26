# {
#  "cells": [
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "## 音声認識モデル"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "import IPython\n",
#     "\n",
#     "app = IPython.Application.instance()\n",
#     "app.kernel.do_shutdown(True)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "import whisper\n",
#     "model = whisper.load_model(\"large-v3\")"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "def transcribe_check_for_words(wave_path, initial_prompt, correct_words):\n",
#     "    # モデルを使って音声を転写\n",
#     "    transcription_result = model.transcribe(\n",
#     "        wave_path,\n",
#     "        verbose=True,\n",
#     "        language='japanese',\n",
#     "        fp16=True,\n",
#     "        without_timestamps=True,\n",
