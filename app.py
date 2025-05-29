import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import time
from io import BytesIO
import random

st.set_page_config(page_title="🧠 AI Image Generator", layout="wide")
st.title("🖼️ AI Image Generator (CPU Version)")

# Use CPU for inference
device = "cpu"

@st.cache_resource(show_spinner=False)
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # Memory optimization for CPU
    return pipe

pipe = load_model()

# Sidebar Inputs
with st.sidebar:
    st.header("🎨 Generate Your Image")
    prompt = st.text_input("Enter prompt", "A futuristic cityscape at sunset")
    num_images = st.slider("Number of images", 1, 4, 1)
    seed_input = st.text_input("Seeds (comma-separated, optional)", "")
    generate_btn = st.button("Generate Images")

# Output Area
progress = st.progress(0)
status = st.empty()
time_text = st.empty()
output_container = st.container()

if generate_btn:
    # Handle seeds
    if seed_input.strip():
        try:
            seeds = [int(s.strip()) for s in seed_input.split(",")]
            if len(seeds) < num_images:
                seeds += [random.randint(0, 2**32 - 1) for _ in range(num_images - len(seeds))]
        except:
            st.error("⚠️ Invalid seed input. Using random seeds.")
            seeds = [random.randint(0, 2**32 - 1) for _ in range(num_images)]
    else:
        seeds = [random.randint(0, 2**32 - 1) for _ in range(num_images)]

    start_time = time.time()
    images = []

    for i in range(num_images):
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1) if i > 0 else 0
        remaining = avg_time * (num_images - i - 1)
        percent = int((i / num_images) * 100)

        progress.progress(percent)
        status.text(f"🖼️ Generating image {i+1}/{num_images} ({percent}%)")
        time_text.text(f"⏳ Estimated time left: {remaining:.1f}s")

        generator = torch.Generator(device).manual_seed(seeds[i])
        image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, generator=generator).images[0]
        images.append((image, seeds[i]))

    progress.progress(100)
    status.text("✅ Done!")
    time_text.text("")

    # Show images
    with output_container:
        cols = st.columns(num_images)
        for idx, (img, seed) in enumerate(images):
            with cols[idx]:
                st.image(img, caption=f"Seed: {seed}", use_column_width=True)
                buf = BytesIO()
                img.save(buf, format="PNG")
                st.download_button("Download PNG", buf.getvalue(), file_name=f"image_{seed}.png", mime="image/png")
