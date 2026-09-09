from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
CNN_MODEL_PATH = PROJECT_ROOT / "models" / "saved_models" / "best_model.h5"

st.set_page_config(
    page_title="Image Classifier",
    page_icon="[ ]",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
    .hero { padding: 2rem 0 1.2rem; }
    .eyebrow { color: #0b7a75; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; font-size: .78rem; }
    .hero h1 { font-size: clamp(2.2rem, 5vw, 4.4rem); line-height: .98; margin: .45rem 0 1rem; color: #102a2a; }
    .hero p { max-width: 650px; color: #52706d; font-size: 1.05rem; }
    .metric { background: #f3f8f5; border-left: 4px solid #0b7a75; padding: 1rem 1.1rem; min-height: 100px; }
    .metric-label { color: #52706d; font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { color: #102a2a; font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; margin-top: .35rem; }
    .note { color: #52706d; font-size: .9rem; padding-top: .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def analyze_image(image: Image.Image):
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    channel_spread = float(np.mean(np.std(rgb.astype(np.float32), axis=2)))
    saturation = float(np.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]))
    is_color = channel_spread > 8.0 or saturation > 20.0

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < 45:
        sharpness = "Blurry"
        sharpness_detail = "Low edge detail"
    elif blur_score < 120:
        sharpness = "Slightly blurry"
        sharpness_detail = "Moderate edge detail"
    else:
        sharpness = "Sharp"
        sharpness_detail = "Good edge detail"

    return {
        "image_type": "Color image" if is_color else "Black and white image",
        "image_type_detail": f"Color variation score: {channel_spread:.1f}",
        "sharpness": sharpness,
        "sharpness_detail": f"Blur score: {blur_score:.1f} ({sharpness_detail})",
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "channels": "RGB",
    }


def predict_flower(image: Image.Image):
    if not CNN_MODEL_PATH.exists():
        return None

    from src.model import create_model

    model = create_model(num_classes=3)
    model.load_weights(str(CNN_MODEL_PATH))
    image_array = np.array(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    predictions = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
    class_index = int(np.argmax(predictions))
    return class_index, float(predictions[class_index])


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Visual inspection workspace</div>
      <h1>Understand every image<br>at a glance.</h1>
      <p>Upload a photo to identify whether it is color or black and white, measure blur, and optionally predict its trained flower category.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, JPEG, PNG, BMP, and WEBP.",
)

if uploaded_file is None:
    st.info("Choose an image above to begin the analysis.")
else:
    try:
        image = Image.open(uploaded_file)
        analysis = analyze_image(image)

        preview, results = st.columns([1.05, 1], gap="large")
        with preview:
            st.image(image, caption=f"{uploaded_file.name}  |  {analysis['width']} x {analysis['height']}")
        with results:
            st.subheader("Image analysis")
            first, second = st.columns(2)
            with first:
                st.markdown(f'<div class="metric"><div class="metric-label">Image type</div><div class="metric-value">{analysis["image_type"]}</div></div>', unsafe_allow_html=True)
            with second:
                st.markdown(f'<div class="metric"><div class="metric-label">Sharpness</div><div class="metric-value">{analysis["sharpness"]}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="note">{analysis["image_type_detail"]}<br>{analysis["sharpness_detail"]}</div>', unsafe_allow_html=True)

            st.subheader("Model prediction")
            prediction = predict_flower(image)
            if prediction is None:
                st.warning("No trained CNN checkpoint found. Image quality analysis is still available.")
            else:
                class_index, confidence = prediction
                st.success(f"Flower class {class_index}  |  confidence {confidence:.1%}")
                st.caption("The trained model predicts flower classes 0, 1, and 2. These are separate from image quality.")

        st.divider()
        st.caption("Color detection uses channel variation. Blur detection uses the variance of the image Laplacian; use the score as a practical guide, not a medical or forensic measurement.")
    except Exception as error:
        st.error(f"Could not analyze this image: {error}")
