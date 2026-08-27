
import re
import whisper
import pykakasi
from moviepy.editor import VideoFileClip

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def to_hiragana(text):
    """
    テキストをひらがなに変換する
    """
    kakasi = pykakasi.kakasi()
    kakasi.setMode("J", "H")  # J(apanese) to H(iragana)
    converter = kakasi.getConverter()
    return converter.do(text)

def remove_punctuation(text):
    """
    テキストから句読点を削除する
    """
    return re.sub(r'[、。,\. ]', '', text)

def convert_video_to_audio(video_path, audio_path):
    """
    ビデオファイルをオーディオファイルに変換
    """
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def whisper_model(model_size='large'):
    """
    whisperモデルのロード
    ・tiny
    ・smart
    ・medium
    ・large（日本語にも対応）
    """
    return whisper.load_model(model_size)

def transcribe_audio(model, audio_path, language='japanese', initial_prompt=''):
    """
    音声データをテキストに書き起こす

    Parameters:
    - model (whisper.Model): Whisperモデルのインスタンス
    - audio_path (str): 書き起こしを行う音声ファイルのパス
    - language (str, optional): 書き起こしを行う言語（japanese）
    - initial_prompt (str, optional): モデルがテキスト生成を開始する前に重要視するテキスト
    このテキストは、書き起こしのコンテキストや指示として機能

    Returns:
    dict: 書き起こしの結果を含む辞書。
    主要なキーには'text'（書き起こされたテキスト）が含まれる

    """
    return model.transcribe(
        audio_path, 
        verbose=True, 
        language=language, 
        #fp16=True, 
        without_timestamps=True, 
        initial_prompt=initial_prompt
    )

def check_words(transcribed_text, words_to_check):
    """
    書き起こされたテキストに特定の単語が含まれているかチェック
    """
    # 句読点を削除し、ひらがなに変換
    cleaned_text = to_hiragana(remove_punctuation(transcribed_text))
    results = {}
    for word in words_to_check:
        # 単語も同様に処理
        cleaned_word = to_hiragana(remove_punctuation(word))
        results[word] = cleaned_word in cleaned_text
        print(f"単語 '{word}' {'True' if results[word] else 'False'}")
    return results


if __name__ == '__main__':
    video_path = '/app/whisper/正面右_4回目.MOV'
    audio_path = '正面右_4回目.wav'
    
    # ビデオをオーディオに変換
    convert_video_to_audio(video_path, audio_path)
    
    # Whisper modelのロード
    model = whisper_model('large')
    
    # 最初のプロンプト & 正解の単語
    initial_prompt = '傷病者 感染防御 頸動脈'
    # correct_words = [
    #     '傷病者', '周囲の安全', '感染防御', '大丈夫ですか', '誰か来てください', 
    #     '1 2 3', '119番', 'AEDを持ってきてください', '呼吸の確認', '脈の確認', '蘇生', '荷物'
    # ]
    correct_words = [
        '周囲の安全よし', '傷病者発見', '感染防御よし','大丈夫ですか','誰か来てください', 'あなた119番通報してください','あなたAEDを持ってきてください',
        '呼吸の確認','脈の確認','1 2 3', 'ショック', '荷物'
    ]
    transcription_result = transcribe_audio(model, audio_path, initial_prompt=initial_prompt)
    transcribed_text = transcription_result['text']
    
    check_words(transcribed_text, correct_words)