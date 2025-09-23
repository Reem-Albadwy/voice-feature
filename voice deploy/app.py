from flask import Flask, request, jsonify, send_from_directory
import os
import subprocess
from processor import run_pipeline

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "tts_table"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files or "text" not in request.form:
        return jsonify({"error": "audio file and text are required"}), 400

    audio_file = request.files["audio"]
    expected_text = request.form["text"]

    opus_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    wav_path = os.path.splitext(opus_path)[0] + ".wav"
    audio_file.save(opus_path)

    # convert OPUS to WAV
    subprocess.run([
        "ffmpeg", "-y", "-i", opus_path,
        "-ar", "24000", "-ac", "1", wav_path
    ], check=False)

    result = run_pipeline(wav_path, expected_text, OUTPUT_FOLDER)
    return jsonify(result)

@app.route("/tts/<filename>")
def serve_tts(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
