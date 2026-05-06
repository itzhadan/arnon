from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import gradio as gr
import os

# 🎤 מודל תמלול
model = WhisperModel(
    "medium",
    compute_type="int8"
)

# 🌍 תמלול + תרגום לעברית
def transcribe(audio):

    if audio is None:
        return "לא נבחר קובץ"

    try:

        # 🧠 זיהוי שפה אוטומטי
        segments, info = model.transcribe(
            audio,
            beam_size=5,
            vad_filter=True
        )

        original_text = " ".join(
            [s.text.strip() for s in segments]
        ).strip()

        if not original_text:
            return "לא זוהה דיבור"

        detected_lang = info.language

        # 🇮🇱 אם זה כבר עברית
        if detected_lang == "he":
            translated = original_text

        else:
            # 🌍 תרגום לעברית
            translated = GoogleTranslator(
                source='auto',
                target='iw'
            ).translate(original_text)

        return f"""
🎧 שפה שזוהתה:
{detected_lang}

📝 תמלול מקורי:
{original_text}

🇮🇱 תרגום לעברית:
{translated}
"""

    except Exception as e:
        return f"שגיאה: {str(e)}"


# 🎨 ממשק יפה
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div style="
        direction: rtl;
        text-align: center;
        padding: 20px;
    ">
        <h1>🎤 תמלול ותרגום הודעות קוליות</h1>

        <p>
        העלה הודעת קול מכל שפה —
        אנגלית, טורקית, רוסית וכו'
        ותקבל תרגום לעברית
        </p>
    </div>
    """)

    audio = gr.Audio(
        type="filepath",
        label="🎧 העלה הודעה קולית"
    )

    output = gr.Textbox(
        lines=15,
        label="📄 תוצאה"
    )

    btn = gr.Button("🚀 תמלל ותרגם")

    btn.click(
        fn=transcribe,
        inputs=audio,
        outputs=output
    )

demo.launch()
