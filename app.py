# ---------------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------------
import json
import io
from pathlib import Path
from datetime import datetime

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ---------------------------------------------------------------------------------
# GLOBAL CONSTANTS & COLOR SCHEME (Emerald Theme)
# ---------------------------------------------------------------------------------
APP_NAME = "LEAF ID"

COLOR_DEEP     = "#047857"
COLOR_MID      = "#10B981"
COLOR_SOFT     = "#34D399"
COLOR_WARN     = "#F59E0B"
COLOR_DANGER   = "#EF4444"

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "braincell_best.pt"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} — Plant Leaf Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------------
# ADAPTIVE CSS (Supports both Light and Dark Mode)
# ---------------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
        h1, h2, h3, h4 {{
            color: {COLOR_DEEP} !important;
        }}

        /* ---------- Hero banner (Stays vibrant in both modes) ---------- */
        .hero-banner {{
            background: linear-gradient(135deg, {COLOR_DEEP} 0%, {COLOR_MID} 100%);
            padding: 60px 40px; /* INCREASED: Adds more space inside the box to make it taller and wider */
            border-radius: 18px;
            margin-bottom: 22px;
            box-shadow: 0 4px 15px rgba(4, 120, 87, 0.3);
        }}
        .hero-title {{
            color: #ffffff !important;
            font-size: 3.8rem; /* INCREASED: Made the LEAF ID text significantly larger (was 2.4rem) */
            font-weight: 800;
            margin: 0;
            letter-spacing: 0.5px;
        }}
        .hero-subtitle {{
            color: #ffffff !important;
            font-size: 1.15rem; /* OPTIONAL: Slightly increased subtitle size to match the bigger box */
            margin-top: 10px;
        }}

        .brand-card {{
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-left: 6px solid {COLOR_MID};
            border-radius: 14px;
            padding: 20px 22px;
            margin-bottom: 16px;
            color: var(--text-color);
        }}

        .brand-badge {{
            display: inline-block;
            background-color: {COLOR_DEEP};
            color: #ffffff !important;
            padding: 6px 18px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1rem;
        }}
        .badge-warning {{
            background-color: {COLOR_DANGER};
        }}
        .badge-healthy {{
            background-color: {COLOR_MID};
        }}

        .brand-divider {{
            border: none;
            height: 2px;
            background: linear-gradient(90deg, {COLOR_DEEP}, transparent);
            border-radius: 2px;
            margin: 14px 0 20px 0;
        }}

        .conf-track, .rank-track {{
            background-color: rgba(128, 128, 128, 0.15);
            border-radius: 10px;
            overflow: hidden;
        }}
        .conf-track {{
            width: 100%; height: 26px; margin: 6px 0 2px 0; border: 1px solid rgba(128, 128, 128, 0.2);
        }}
        .rank-track {{
            flex-grow: 1; height: 20px; border: 1px solid rgba(128, 128, 128, 0.2);
        }}
        .conf-fill, .rank-fill {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white !important;
            font-weight: 700;
            font-size: 0.85rem;
            transition: width 0.6s ease-in-out;
        }}

        .rank-row {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}
        .rank-label {{
            width: 230px;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-color);
            padding-right: 10px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .step-card {{
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 14px;
            padding: 18px;
            text-align: center;
            height: 100%;
            color: var(--text-color);
        }}
        .step-number {{
            display: inline-block;
            background-color: {COLOR_DEEP};
            color: white !important;
            width: 34px;
            height: 34px;
            border-radius: 50%;
            line-height: 34px;
            font-weight: 800;
            margin-bottom: 8px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------------
# HERO BANNER
# ---------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-banner">
        <p class="hero-title">{APP_NAME}</p>
        <p class="hero-subtitle">
            Point your camera at a leaf. LEAF ID tells you the species — and, for
            select species, whether it's showing signs of disease.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------------
# MODEL / DATA LOADING (EfficientNet-B0)
# ---------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading class labels...")
def load_class_names(path: Path):
    if not path.exists():
        st.error(f"Could not find class_names.json at: {path}")
        st.stop()
    with open(path, "r") as f:
        names = json.load(f)
    return names

@st.cache_resource(show_spinner="Waking up EfficientNet-B0...")
def load_model(model_path: Path, num_classes: int):
    if not model_path.exists():
        st.error(f"Could not find model weights at: {model_path}")
        st.stop()

    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    state_dict = torch.load(model_path, map_location=DEVICE)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# ---------------------------------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------------------------------
preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]
)

def predict(model, image: Image.Image, class_names: list, top_k: int = 3):
    image = image.convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_probs, top_idxs = torch.topk(probabilities, k=min(top_k, len(class_names)))
    return [
        (class_names[idx], float(prob) * 100.0)
        for prob, idx in zip(top_probs, top_idxs)
    ]

def format_class_name(raw_name: str) -> str:
    is_diseased = "diseased" in raw_name.lower()
    cleaned = raw_name.replace("_diseased", "").replace("-", " ").replace("_", " ")
    tokens = cleaned.split()
    if tokens and len(tokens[-1]) <= 4 and tokens[-1].lower().startswith("p") and tokens[-1][1:].isalnum():
        tokens = tokens[:-1]
    cleaned = " ".join(tokens).strip().title()
    return f"{cleaned} (Diseased)" if is_diseased else cleaned

def confidence_style(pct: float):
    if pct >= 85:
        return "High Confidence", COLOR_MID
    elif pct >= 50:
        return "Moderate Confidence", COLOR_WARN
    else:
        return "Low Confidence", COLOR_DANGER

# ---------------------------------------------------------------------------------
# LOAD CLASS NAMES + MODEL (cached, runs once)
# ---------------------------------------------------------------------------------
class_names = load_class_names(CLASS_NAMES_PATH)
model = load_model(MODEL_PATH, num_classes=len(class_names))

diseased_labels = [c for c in class_names if "diseased" in c.lower()]
species_only_labels = [c for c in class_names if c not in diseased_labels]
disease_capable_species = sorted({format_class_name(c).replace(" (Diseased)", "") for c in diseased_labels})

# ---------------------------------------------------------------------------------
# SIDEBAR — About the Project
# ---------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<h2 style='color:{COLOR_DEEP};'>About {APP_NAME}</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        **LEAF ID** is a deep-learning leaf recognition tool built on an **EfficientNet-B0** architecture. It was trained across a wide, real-world collection of leaves — 
        from common fruit trees and garden vegetables to traditional medicinal and 
        ornamental plants — so it can tell one species from another at a glance, 
        and catch early signs of disease.
        """
    )

    st.markdown("<hr class='brand-divider'>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{COLOR_DEEP};'>Dataset at a Glance</h4>", unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    m1.metric("Total Classes", len(class_names))
    m2.metric("Disease Labels", len(diseased_labels))

    st.caption(
        f"Includes species like *aloevera, mango, guava, tomato, turmeric, "
        f"neem, hibiscus* and **{len(class_names) - 1}+** others."
    )

    with st.expander("Species with disease detection"):
        st.write(
            "For the species below, LEAF ID can distinguish a **healthy** leaf "
            "from a **diseased** one:"
        )
        for sp in disease_capable_species:
            st.markdown(f"- {sp}")

    st.markdown("<hr class='brand-divider'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        **Model Architecture:** EfficientNet-B0 
        **Input size:** 224 × 224 px
        **Framework:** PyTorch + Streamlit
        **Runs on:** CPU only — 100% offline capable
        """
    )
    st.caption(f"Session started: {datetime.now().strftime('%b %d, %Y — %H:%M')}")

# ---------------------------------------------------------------------------------
# MAIN AREA — TABS
# ---------------------------------------------------------------------------------
tab_diagnose, tab_insights, tab_how = st.tabs(
    ["Diagnose a Leaf", "Dataset Insights", "How It Works"]
)

# =========================== TAB 1 — DIAGNOSE =====================================
with tab_diagnose:
    st.markdown(
        f"<h3 style='color:{COLOR_DEEP};'>Upload a Leaf Photo</h3>",
        unsafe_allow_html=True,
    )
    st.write("For best results, use a well-lit, close-up photo of a single leaf against a plain background.")

    uploaded_file = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"], key="uploader")

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        left, right = st.columns([1, 1.3])

        with left:
            st.markdown(f"<h4 style='color:{COLOR_DEEP};'>Your Image</h4>", unsafe_allow_html=True)
            st.image(image, caption="Uploaded Leaf", use_container_width=True)

        with st.spinner("Running the leaf through EfficientNet-B0..."):
            top_predictions = predict(model, image, class_names, top_k=3)

        top_raw, top_confidence = top_predictions[0]
        top_display = format_class_name(top_raw)
        is_diseased = "diseased" in top_raw.lower()
        conf_label, conf_color = confidence_style(top_confidence)

        with right:
            st.markdown(f"<h4 style='color:{COLOR_DEEP};'>Diagnosis</h4>", unsafe_allow_html=True)

            badge_class = "badge-warning" if is_diseased else "badge-healthy"
            
            st.markdown(
                f"""
                <div class="brand-card">
                    <span class="brand-badge {badge_class}">
                        {top_display}
                    </span>
                    <div class="conf-track">
                        <div class="conf-fill" style="width:{top_confidence:.1f}%; background-color:{conf_color};">
                            {top_confidence:.1f}%
                        </div>
                    </div>
                    <p style="margin-top:6px; color:var(--text-color); font-size:0.9rem;">
                        <b>{conf_label}</b> in this prediction
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if is_diseased:
                st.warning(
                    f"Signs consistent with **{top_display}** were detected. "
                    "Consider isolating the plant and consulting an agricultural expert."
                )
            else:
                st.success(f"This leaf looks like a healthy **{top_display}** leaf.")
                if top_confidence >= 90:
                    st.balloons()

        st.markdown("<hr class='brand-divider'>", unsafe_allow_html=True)

        st.markdown(f"<h3 style='color:{COLOR_DEEP};'>Top 3 Predictions</h3>", unsafe_allow_html=True)

        rank_colors = [COLOR_MID, COLOR_WARN, COLOR_SOFT]
        for i, (raw_name, conf) in enumerate(top_predictions):
            label = format_class_name(raw_name)
            st.markdown(
                f"""
                <div class="rank-row">
                    <div class="rank-label">#{i+1} {label}</div>
                    <div class="rank-track">
                        <div class="rank-fill" style="width:{conf:.1f}%; background-color:{rank_colors[i % 3]};">
                            {conf:.1f}%
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("View as a table"):
            df = pd.DataFrame(
                [(format_class_name(n), round(c, 2)) for n, c in top_predictions],
                columns=["Prediction", "Confidence (%)"],
            )
            st.dataframe(df, hide_index=True, use_container_width=True)

    else:
        st.info("Upload a leaf image above to get started.")
        st.markdown(
            f"""
            <div class="brand-card">
                <b>Tip:</b> LEAF ID recognizes {len(class_names)} classes — from
                common crops like tomato and corn, to medicinal plants like aloevera, neem, and basil.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================== TAB 2 — DATASET INSIGHTS =============================
with tab_insights:
    st.markdown(f"<h3 style='color:{COLOR_DEEP};'>What Can {APP_NAME} Recognize?</h3>", unsafe_allow_html=True)
    st.write(
        f"LEAF ID's {len(class_names)}-class vocabulary spans fruit trees, garden "
        "vegetables, spices, and traditional medicinal or ornamental plants."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Classes", len(class_names))
    c2.metric("Species w/ Disease Pairs", len(disease_capable_species))
    c3.metric("Species-only Labels", len(species_only_labels) - len(disease_capable_species) * 0)

    st.markdown("<hr class='brand-divider'>", unsafe_allow_html=True)

    comp_df = pd.DataFrame(
        {
            "Category": ["Healthy / Species-only Labels", "Diseased Labels"],
            "Count": [len(class_names) - len(diseased_labels), len(diseased_labels)],
        }
    ).set_index("Category")
    st.markdown(f"<h4 style='color:{COLOR_DEEP};'>Label Composition</h4>", unsafe_allow_html=True)
    st.bar_chart(comp_df, color=COLOR_MID)

    st.markdown(f"<h4 style='color:{COLOR_DEEP};'>Browse All Classes</h4>", unsafe_allow_html=True)
    search = st.text_input("Filter classes", placeholder="e.g. mango, diseased, herb...")

    display_names = sorted(format_class_name(c) for c in class_names)
    if search:
        display_names = [n for n in display_names if search.lower() in n.lower()]

    cols = st.columns(3)
    for i, name in enumerate(display_names):
        with cols[i % 3]:
            tag_color = COLOR_DANGER if "(Diseased)" in name else COLOR_MID
            st.markdown(
                f"""<span style="display:inline-block; background-color:{tag_color};
                color:white !important; padding:4px 10px; border-radius:12px; font-size:0.8rem;
                margin-bottom:6px;">{name}</span>""",
                unsafe_allow_html=True,
            )

# =========================== TAB 3 — HOW IT WORKS ================================
with tab_how:
    st.markdown(f"<h3 style='color:{COLOR_DEEP};'>How {APP_NAME} Works</h3>", unsafe_allow_html=True)
    st.write("From your photo to a diagnosis, in four steps:")

    steps = [
        ("1", "Capture", "You upload a clear photo of a single leaf."),
        ("2", "Preprocess", "The image is resized to 224×224 and normalized."),
        ("3", "Inference", "EfficientNet-B0 scores every class."),
        ("4", "Report", "The top-3 most likely classes are ranked and color-coded."),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-number">{num}</div>
                    <br>
                    <b>{title}</b>
                    <p style="font-size:0.85rem; color:var(--text-color); opacity: 0.8; margin-top:5px;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='brand-divider'>", unsafe_allow_html=True)

    st.markdown(f"<h4 style='color:{COLOR_DEEP};'>Under the Hood</h4>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="brand-card">
        <b>Architecture:</b> EfficientNet-B0 backbone with a custom linear
        classifier head sized to {len(class_names)} output classes.<br><br>
        <b>Preprocessing pipeline:</b> RGB conversion → resize to 224×224 →
        tensor conversion → normalization with ImageNet mean
        <code>[0.485, 0.456, 0.406]</code> and std
        <code>[0.229, 0.224, 0.225]</code>.<br><br>
        <b>Inference:</b> Softmax over the model's output logits converts raw
        scores into probabilities, from which the top-3 classes are shown.<br><br>
        <b>Deployment:</b> The model and all computation are forced onto the CPU,
        so LEAF ID runs identically on a laptop with no GPU. It is very Light-weight.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------------
st.markdown("<hr class='brand-divider'>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align:center; color:var(--text-color); opacity:0.6; font-size:0.85rem;'>"
    f"{APP_NAME} - Made with Love and a lot of effort"
    "</p>",
    unsafe_allow_html=True,
)