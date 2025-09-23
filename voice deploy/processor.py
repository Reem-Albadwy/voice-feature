import os, re, logging, torch
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from TTS.api import TTS
import pandas as pd

def normalize_ar(text):
    text = text.lower()
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"[^ء-ي0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def words_list(text):
    return normalize_ar(text).split()

def run_pipeline(wav_path, expected_text, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ASR
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large", device=0 if device=="cuda" else -1)
    spoken_text = asr(wav_path)["text"]

    # Diacritizer
    model_name = "glonor/byt5-arabic-diacritization"
    tokenizer_diac = AutoTokenizer.from_pretrained(model_name)
    model_diac = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def diacritize_sentence(sent, max_out=256):
        inputs = tokenizer_diac(sent, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model_diac.generate(**inputs, max_new_tokens=max_out, num_beams=5, early_stopping=True)
        return tokenizer_diac.decode(outputs[0], skip_special_tokens=True)

    expected_diac = diacritize_sentence(expected_text)

    # Compare
    matcher = SequenceMatcher(None, words_list(expected_text), words_list(spoken_text))
    substitutions, deletions = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            substitutions.append((expected_text.split()[i1:i2], spoken_text.split()[j1:j2]))
        elif tag == "delete":
            deletions.extend(expected_text.split()[i1:i2])

    # TTS
    logging.getLogger("TTS").setLevel(logging.CRITICAL)
    os.environ["COQUI_TTS_LOG_LEVEL"] = "ERROR"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for idx, (correct, wrong) in enumerate(substitutions):
        correct_txt = " ".join(correct)
        out_path = os.path.join(out_dir, f"sub_{idx}.wav")
        tts.tts_to_file(text=correct_txt, file_path=out_path, language="ar", speaker_wav=wav_path)
        rows.append({"correct": correct_txt, "wrong": " ".join(wrong), "tts_file": f"/tts/sub_{idx}.wav"})

    for j, word in enumerate(deletions, start=len(substitutions)):
        out_path = os.path.join(out_dir, f"del_{j}.wav")
        tts.tts_to_file(text=word, file_path=out_path, language="ar", speaker_wav=wav_path)
        rows.append({"correct": word, "wrong": "لم تُذكر", "tts_file": f"/tts/del_{j}.wav"})

    return {
        "expected_text": expected_text,
        "spoken_text": spoken_text,
        "expected_diac": expected_diac,
        "differences": rows
    }
