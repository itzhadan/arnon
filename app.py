from flask import Flask, request, render_template_string
import subprocess
from faster_whisper import WhisperModel

app = Flask(__name__)

# מודל – התחל עם medium, ואם המחשב סוחב תעבור ל-large-v3
model = WhisperModel("medium", compute_type="int8")

HTML = """
<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<title>תמלול קול</title>
</head>
<body>
<h2>🎤 העלה קובץ קול</h2>

<form method="post" action="/transcribe" enctype="multipart/form-data">
    <input type="file" name="file" required>
    <button type="submit">שלח</button>
</form>

{% if result %}
<h3>📝 תוצאה:</h3>
<p>{{ result }}</p>
{% endif %}

{% if error %}
<h3 style="color:red;">❌ שגיאה:</h3>
<p>{{ error }}</p>
{% endif %}
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        file = request.files["file"]
        input_file = "input.opus"
        wav_file = "clean.wav"

        file.save(input_file)

        # המרה ל-WAV תקין
        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_file,
            "-ar", "16000",
            "-ac", "1",
            wav_file
        ], check=True)

        # תמלול (עברית)
        segments, _ = model.transcribe(wav_file, language="he")
        text = " ".join([s.text for s in segments])

        return render_template_string(HTML, result=text)

    except Exception as e:
        return render_template_string(HTML, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
