import gradio as gr
import torch
import tempfile
import os
import datetime
import hashlib

from heartlib import HeartMuLaGenPipeline

# -----------------------------
# Model loading (once)
# -----------------------------
PIPELINE = None

def load_pipeline(model_path, version):
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = HeartMuLaGenPipeline.from_pretrained(
            model_path,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            version=version,
        )
    return PIPELINE


# -----------------------------
# Generation function
# -----------------------------
def generate_music(
    model_path,
    version,
    lyrics_text,
    tags_text,
    max_audio_length_ms,
    topk,
    temperature,
    cfg_scale,
):
    pipe = load_pipeline(model_path, version)

    # -----------------------------
    # Output filename logic
    # -----------------------------
    os.makedirs("outputs", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    content_hash = hashlib.sha1(
        (lyrics_text + tags_text).encode("utf-8")
    ).hexdigest()[:6]

    filename = f"heartmula_{timestamp}_{content_hash}.mp3"
    output_path = os.path.join("outputs", filename)

    # -----------------------------
    # Generation
    # -----------------------------
    with torch.no_grad():
        pipe(
            {
                "lyrics": lyrics_text,
                "tags": tags_text,
            },
            max_audio_length_ms=max_audio_length_ms,
            save_path=output_path,
            topk=topk,
            temperature=temperature,
            cfg_scale=cfg_scale,
        )

    return output_path


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="HeartMuLaGen Music Generator") as demo:
    gr.Markdown("## 🎵 HeartMuLaGen Music Generator")

    with gr.Row():
        model_path = gr.Textbox(
            label="Model Path",
            placeholder="/path/to/model",
            value="",
        )
        version = gr.Textbox(
            label="Model Version",
            value="3B",
        )

    lyrics = gr.Textbox(
        label="Lyrics",
        placeholder="Enter lyrics here...",
        lines=8,
    )

    tags = gr.Textbox(
        label="Tags / Prompt",
        placeholder="e.g. pop, emotional, female vocal",
        lines=4,
    )

    with gr.Row():
        max_audio_length_ms = gr.Slider(
            10_000, 600_000, value=240_000, step=1_000,
            label="Max Audio Length (ms)"
        )
        topk = gr.Slider(1, 200, value=50, step=1, label="Top-k")
        temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
        cfg_scale = gr.Slider(0.1, 5.0, value=1.5, step=0.1, label="CFG Scale")

    generate_btn = gr.Button("🎶 Generate Music")

    output_audio = gr.Audio(
        label="Generated Audio",
        type="filepath",
        interactive=False,
    )

    generate_btn.click(
        fn=generate_music,
        inputs=[
            model_path,
            version,
            lyrics,
            tags,
            max_audio_length_ms,
            topk,
            temperature,
            cfg_scale,
        ],
        outputs=output_audio,
    )


if __name__ == "__main__":
    demo.launch()