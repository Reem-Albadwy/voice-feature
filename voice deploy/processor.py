# voice deploy/processor.py
import os, re, logging, torch
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from TTS.api import TTS

# ======================
# Text normalization
# ======================
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

# ======================
# Mapping functions
# ======================
def build_orig_index_map(original_text):
    orig_words = original_text.split()
    norm_words = words_list(original_text)
    mapping = {}
    j = 0
    for i, ow in enumerate(orig_words):
        if j < len(norm_words) and normalize_ar(ow) == norm_words[j]:
            mapping[j] = {"orig": ow, "diac": None}
            j += 1
    return mapping

def build_mapping_with_diac(original_text, diac_text):
    orig_words = original_text.split()
    norm_orig = words_list(original_text)

    diac_tokens = diac_text.split()
    norm_diac = [normalize_ar(t) for t in diac_tokens]

    mapping = {}
    i = j = 0
    while i < len(norm_orig) and j < len(norm_diac):
        if norm_orig[i] == norm_diac[j]:
            mapping[i] = {"orig": orig_words[i], "diac": diac_tokens[j]}
            i += 1; j += 1
        else:
            found = False
            for k in range(1, 4):
                if j + k < len(norm_diac) and norm_orig[i] == "".join(norm_diac[j:j+k]):
                    mapping[i] = {
                        "orig": orig_words[i],
                        "diac": " ".join(diac_tokens[j:j+k])
                    }
                    j += k
                    i += 1
                    found = True
                    break
            if not found:
                i += 1

    # لو كلمة مش متغطية
    for idx, ow in enumerate(orig_words):
        if idx not in mapping:
            mapping[idx] = {"orig": ow, "diac": ow}
    return mapping

# ======================
# Sentence splitting
# ======================
def split_sentences_preserve(text):
    pieces = re.split(r'([.؟!\n]|،)', text)
    sentences, cur = [], ""
    for p in pieces:
        if p is None:
            continue
        cur += p
        if re.match(r'[.؟!\n]|،', p):
            if cur.strip():
                sentences.append(cur.strip())
            cur = ""
    if cur.strip():
        sentences.append(cur.strip())
    return [s for s in sentences if s.strip()]

# ======================
# Compare
# ======================
def compare_words(expected_text, spoken_text):
    exp_norm = words_list(expected_text)
    spk_norm = words_list(spoken_text)

    exp_map = build_orig_index_map(expected_text)
    spk_map = build_orig_index_map(spoken_text)

    matcher = SequenceMatcher(None, exp_norm, spk_norm)
    substitutions, deletions = [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            exp_seg = [exp_map.get(i, {"orig": exp_norm[i]})["orig"] for i in range(i1, i2)]
            spk_seg = [spk_map.get(j, {"orig": spk_norm[j]})["orig"] for j in range(j1, j2)]
            substitutions.append((" ".join(exp_seg), " ".join(spk_seg)))
        elif tag == "delete":
            for i in range(i1, i2):
                deletions.append(exp_map.get(i, {"orig": exp_norm[i]})["orig"])

    return {"substitutions": substitutions, "deletions": deletions}

# ======================
# Run pipeline
# ======================
def run_pipeline(wav_path, expected_text, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- ASR ---
    asr = pipeline("automatic-speech-recognition",
                   model="openai/whisper-large",
                   device=0 if device == "cuda" else -1)
    spoken_text = asr(wav_path)["text"]

    # --- Diacritizer ---
    model_name = "glonor/byt5-arabic-diacritization"
    tokenizer_diac = AutoTokenizer.from_pretrained(model_name)
    model_diac = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def diacritize_sentence(sent, max_out=256):
        inputs = tokenizer_diac(sent, return_tensors="pt",
                                truncation=True, padding=True).to(device)
        outputs = model_diac.generate(**inputs,
                                      max_new_tokens=max_out,
                                      num_beams=5,
                                      early_stopping=True)
        diac = tokenizer_diac.decode(outputs[0],
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        # إزالة التكرارات
        diac = re.sub(r'([ًٌٍَُِْ])\1+', r'\1', diac)
        toks = diac.split()
        cleaned = []
        for t in toks:
            if len(cleaned) >= 1 and t == cleaned[-1]:
                continue
            cleaned.append(t)
        return " ".join(cleaned)

    expected_diac_sents = []
    for s in split_sentences_preserve(expected_text):
        try:
            expected_diac_sents.append(diacritize_sentence(s, max_out=512))
        except Exception:
            expected_diac_sents.append(s)
    expected_diac = " ".join(expected_diac_sents)

    # --- Mapping ---
    exp_map_with_diac = build_mapping_with_diac(expected_text, expected_diac)

    # --- Compare ---
    res = compare_words(expected_text, spoken_text)

    # --- TTS ---
    logging.getLogger("TTS").setLevel(logging.CRITICAL)
    os.environ["COQUI_TTS_LOG_LEVEL"] = "ERROR"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    os.makedirs(out_dir, exist_ok=True)
    rows = []

    # substitutions
    for i, (correct, wrong) in enumerate(res["substitutions"]):
        correct_diac = []
        for token in correct.split():
            found = False
            for t in expected_diac.split():
                if normalize_ar(t) == normalize_ar(token):
                    correct_diac.append(t)
                    found = True
                    break
            if not found:
                correct_diac.append(token)
        correct_diac_txt = " ".join(correct_diac)

        out_path = os.path.join(out_dir, f"sub_{i}.wav")
        tts.tts_to_file(text=correct_diac_txt,
                        file_path=out_path,
                        language="ar",
                        speaker_wav=wav_path)

        rows.append({
            "correct": correct,
            "wrong": wrong,
            "correct_diac": correct_diac_txt,
            "tts_file": f"/tts/sub_{i}.wav"
        })

    # deletions
    for j, word in enumerate(res["deletions"], start=len(res["substitutions"])):
        # حاول تجيبها من الـ diac
        word_diac = word
        for t in expected_diac.split():
            if normalize_ar(t) == normalize_ar(word):
                word_diac = t
                break

        out_path = os.path.join(out_dir, f"del_{j}.wav")
        tts.tts_to_file(text=word_diac,
                        file_path=out_path,
                        language="ar",
                        speaker_wav=wav_path)

        rows.append({
            "correct": word,
            "wrong": "لم تُذكر",
            "correct_diac": word_diac,
            "tts_file": f"/tts/del_{j}.wav"
        })

    return {
        "expected_text": expected_text,
        "spoken_text": spoken_text,
        "expected_diac": expected_diac,
        "mapping_with_diac": exp_map_with_diac,
        "differences": rows
    }
