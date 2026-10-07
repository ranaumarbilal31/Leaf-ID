# =================================================================================
# LEAF ID Pro — Intelligent Plant Leaf & Pathology Classifier
# =================================================================================
import os
import io
import json
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------------
st.set_page_config(
    page_title="LEAF ID Pro — Plant Species & Disease AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------------
# CONSTANTS & ASSET PATHS
# ---------------------------------------------------------------------------------
APP_NAME = "LEAF ID Pro"
BASE_DIR = Path(__file__).resolve().parent

# Check candidate model weight filenames
MODEL_CANDIDATES = [
    BASE_DIR / "LeafID.pt",
    BASE_DIR / "braincell_best.pt",
    BASE_DIR / "LeadID.pt",
]
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
SAMPLES_DIR = BASE_DIR / "samples"

IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------------
# COMPREHENSIVE BOTANICAL & PATHOLOGY KNOWLEDGE BASE
# ---------------------------------------------------------------------------------
BOTANICAL_ADVICE = {
    "lemon": {
        "diseased": {
            "condition": "Citrus Canker / Bacterial Blight / Scab",
            "symptoms": "Raised corky lesions surrounded by oily or yellow chlorotic halos on foliage and twigs.",
            "cause": "Xanthomonas citri bacteria spread by wind-driven rain, irrigation splashing, or leaf miners.",
            "treatment": [
                "Prune severely affected twigs and destroy fallen infected leaves.",
                "Apply copper-based bactericide/fungicide spray during early leaf flush.",
                "Avoid overhead irrigation to minimize moisture on foliage.",
                "Control citrus leaf miner larvae which create entry wounds for infection."
            ],
            "severity": "Moderate to High"
        },
        "healthy": {
            "care": "Citrus limon thrives in full sunlight (6-8 hours daily), well-draining loamy soil with pH 5.5-6.5, and regular deep watering with drying intervals."
        }
    },
    "mango": {
        "diseased": {
            "condition": "Anthracnose / Leaf Blight",
            "symptoms": "Dark brown, angular to irregular necrotic spots that coalesce into large necrotic blights.",
            "cause": "Colletotrichum gloeosporioides fungus prevalent in high humidity and warm temperatures.",
            "treatment": [
                "Prune canopy to improve air circulation and sunlight penetration.",
                "Spray neem oil or copper oxychloride fungicide every 14 days during wet spells.",
                "Remove and dispose of diseased twigs and fallen debris around the root zone."
            ],
            "severity": "Moderate"
        },
        "healthy": {
            "care": "Mangifera indica prefers warm tropical/subtropical climate, deep well-drained soil, and deep infrequent watering once established."
        }
    },
    "tomato": {
        "diseased": {
            "condition": "Early Blight / Septoria Leaf Spot",
            "symptoms": "Concentric target-like brown spots with yellow halos, starting on lower leaves and moving upwards.",
            "cause": "Alternaria solani / Septoria lycopersici fungi thriving in humid, warm conditions.",
            "treatment": [
                "Remove bottom foliage within 12 inches of the soil to prevent soil-splash contamination.",
                "Apply organic copper fungicide or bio-fungicide (Bacillus subtilis).",
                "Mulch heavily around base with straw to create a barrier over soil pathogens.",
                "Water strictly at base level using drip or soaker hoses."
            ],
            "severity": "High"
        },
        "healthy": {
            "care": "Solanum lycopersicum needs fertile, compost-rich soil, constant moisture, staking support, and 8+ hours of direct sun."
        }
    },
    "guava": {
        "diseased": {
            "condition": "Guava Wilt / Anthracnose",
            "symptoms": "Curling, yellowing leaves with rusty-brown spots leading to premature leaf defoliation.",
            "cause": "Fusarium oxysporum f. sp. psidii or Colletotrichum fungal pathogens.",
            "treatment": [
                "Drench root zone with bio-control agents like Trichoderma viride.",
                "Apply balanced organic fertilizer enriched with zinc and boron.",
                "Ensure soil is free from waterlogging and root nematodes."
            ],
            "severity": "High"
        },
        "healthy": {
            "care": "Psidium guajava is resilient and adaptable to varied soils, tolerating drought once mature and flourishing in sunny conditions."
        }
    },
    "pomegranate": {
        "diseased": {
            "condition": "Bacterial Blight / Cercospora Spot",
            "symptoms": "Water-soaked dark brown spots that turn black with chlorotic margins; cracked petioles.",
            "cause": "Xanthomonas axonopodis pv. punicae or Cercospora punicae.",
            "treatment": [
                "Prune during dry periods and sterilize pruning tools between cuts.",
                "Spray copper hydroxide combined with streptomycin sulphate as recommended by agricultural extensions.",
                "Maintain weed-free orchard floor to lower ambient micro-humidity."
            ],
            "severity": "High"
        },
        "healthy": {
            "care": "Punica granatum loves Mediterranean climates, sunny locations, well-drained gravelly soil, and light pruning to encourage fruiting branches."
        }
    },
    "jamun": {
        "diseased": {
            "condition": "Leaf Spot / Anthracnose",
            "symptoms": "Small circular reddish-brown specks that expand into ragged holes (shot-hole appearance).",
            "cause": "Glomerella cingulata or Cercospora eugeniae fungi.",
            "treatment": [
                "Spray Bordeaux mixture (1%) or systemic fungicide at onset of symptoms.",
                "Collect and burn fallen infested foliage to break the spore lifecycle."
            ],
            "severity": "Moderate"
        },
        "healthy": {
            "care": "Syzygium cumini (Black Plum) is a sturdy evergreen tropical tree with medicinal bark and leaves, needing minimal upkeep once established."
        }
    },
    "jatropha": {
        "diseased": {
            "condition": "Powdery Mildew / Rust",
            "symptoms": "White powdery fungal coating on leaf upper surfaces causing distortion and yellowing.",
            "cause": "Oidium caricae or Phakopsora jatrophicola.",
            "treatment": [
                "Apply potassium bicarbonate or wettable sulfur spray.",
                "Thin branches to increase interior air flow."
            ],
            "severity": "Low to Moderate"
        },
        "healthy": {
            "care": "Jatropha curcas is drought-resistant, thrives in poor sandy soils, and is prized for biofuel seeds and boundary fencing."
        }
    },
    "alstonia_scholaris": {
        "diseased": {
            "condition": "Leaf Gall / Insect Blister",
            "symptoms": "Raised blister-like gall formations on the upper leaf surface caused by psyllid gall midges.",
            "cause": "Pauropsylla tuberculata infestation stimulating abnormal plant tissue growth.",
            "treatment": [
                "Prune heavily galled leaves before insects emerge.",
                "Apply neem seed kernel extract (5%) spray during new flush emergence."
            ],
            "severity": "Low to Moderate"
        },
        "healthy": {
            "care": "Alstonia scholaris (Devil Tree / Saptaparni) is an ornamental shade tree with whorled leaves and aromatic winter blossoms."
        }
    },
    "chinar": {
        "diseased": {
            "condition": "Sycamore Anthracnose",
            "symptoms": "Brown necrotic lesions following leaf veins; sudden blight of young spring foliage.",
            "cause": "Apiognomonia veneta fungal pathogen active in cool, wet spring weather.",
            "treatment": [
                "Rake and destroy fallen leaves in autumn.",
                "Apply systemic fungicide micro-injections for heritage trees if recurring heavily."
            ],
            "severity": "Moderate"
        },
        "healthy": {
            "care": "Platanus orientalis (Oriental Plane) is a majestic long-lived shade tree common in temperate valleys, preferring deep moist soils."
        }
    },
    "pongamia_pinnata": {
        "diseased": {
            "condition": "Tar Spot / Gall Mite Blight",
            "symptoms": "Black tar-like raised fungal spots or blistered galls across the lamina.",
            "cause": "Phyllachora pongamiae or Eriophyid mites.",
            "treatment": [
                "Apply wettable sulfur or horticultural oil during early spring.",
                "Clear leaf litter beneath tree canopy."
            ],
            "severity": "Low to Moderate"
        },
        "healthy": {
            "care": "Millettia pinnata (Karanja) is a nitrogen-fixing hardy tree known for insecticidal seed oil and tolerance of saline and waterlogged soils."
        }
    },
    "arjun": {
        "diseased": {
            "condition": "Leaf Rust / Gall Formation",
            "symptoms": "Orange-yellowish pustules on leaf underside or irregular swelling on veins.",
            "cause": "Puccinia spp. or gall midges.",
            "treatment": [
                "Prune isolated affected leaves.",
                "Apply bio-fungicide or copper oxychloride if rust spreads."
            ],
            "severity": "Low to Moderate"
        },
        "healthy": {
            "care": "Terminalia arjuna is an Ayurvedic medicinal tree native to riverbanks, with cardiovascular benefits derived from its bark."
        }
    }
}

# ---------------------------------------------------------------------------------
# CUSTOM MODERN STYLING (Responsive Botanical Dark/Light Theme)
# ---------------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Import clean typography */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Hero Container */
        .hero-wrapper {
            background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 20px;
            padding: 38px 36px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px -10px rgba(4, 120, 87, 0.45);
            position: relative;
            overflow: hidden;
        }
        .hero-wrapper::after {
            content: "🌿";
            font-size: 140px;
            position: absolute;
            right: 20px;
            bottom: -30px;
            opacity: 0.12;
            pointer-events: none;
        }
        .hero-title {
            color: #ffffff !important;
            font-size: 2.8rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0 0 10px 0;
            letter-spacing: -0.02em;
        }
        .hero-tagline {
            color: #d1fae5 !important;
            font-size: 1.1rem;
            font-weight: 400;
            max-width: 720px;
            margin-bottom: 18px;
            line-height: 1.5;
        }
        .hero-badges-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.14);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.22);
            color: #ffffff !important;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 5px 14px;
            border-radius: 9999px;
        }

        /* Glassmorphism Cards */
        .bio-card {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 16px;
            padding: 22px;
            margin-bottom: 18px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            transition: all 0.2s ease;
        }
        .bio-card:hover {
            border-color: rgba(16, 185, 129, 0.35);
        }

        /* Status & Alert Badges */
        .verdict-banner {
            border-radius: 14px;
            padding: 18px 22px;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-weight: 600;
        }
        .verdict-healthy {
            background: linear-gradient(90deg, rgba(16, 185, 129, 0.18) 0%, rgba(5, 150, 105, 0.08) 100%);
            border: 1px solid #10b981;
            color: #10b981;
        }
        .verdict-diseased {
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.18) 0%, rgba(220, 38, 38, 0.08) 100%);
            border: 1px solid #ef4444;
            color: #ef4444;
        }
        .verdict-inconclusive {
            background: linear-gradient(90deg, rgba(245, 158, 11, 0.18) 0%, rgba(217, 119, 6, 0.08) 100%);
            border: 1px solid #f59e0b;
            color: #f59e0b;
        }

        /* Ranking bar styling */
        .pred-bar-container {
            margin-bottom: 12px;
        }
        .pred-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .pred-track {
            background: rgba(128, 128, 128, 0.15);
            border-radius: 9999px;
            height: 12px;
            overflow: hidden;
            width: 100%;
        }
        .pred-fill {
            height: 100%;
            border-radius: 9999px;
            transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Botanical Tags */
        .plant-tag {
            display: inline-block;
            background: rgba(16, 185, 129, 0.12);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.28);
            border-radius: 8px;
            padding: 5px 12px;
            font-size: 0.82rem;
            font-weight: 600;
            margin: 4px;
        }
        .plant-tag-diseased {
            background: rgba(239, 68, 68, 0.12);
            color: #ef4444;
            border-color: rgba(239, 68, 68, 0.28);
        }

        /* Pipeline Step Card */
        .step-box {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 14px;
            padding: 20px 16px;
            text-align: center;
            height: 100%;
        }
        .step-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #10b981;
            color: #ffffff;
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------------------
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

# ---------------------------------------------------------------------------------
# DATA & MODEL LOADING FUNCTIONS
# ---------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Reading botanical taxonomy classes...")
def load_class_names(path: Path):
    if not path.exists():
        st.error(f"Missing class registry at: {path}")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner="Initializing EfficientNet-B0 inference engine...")
def load_model(candidates: list, num_classes: int):
    # Find existing model path
    active_path = None
    for candidate in candidates:
        if candidate.exists():
            active_path = candidate
            break

    if active_path is None:
        st.error(
            f"❌ Model weights not found! Checked locations:\n" +
            "\n".join([f"- `{c}`" for c in candidates]) +
            "\n\nPlease ensure `LeafID.pt` exists in the application root directory."
        )
        st.stop()

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    try:
        checkpoint = torch.load(active_path, map_location=DEVICE)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Error loading model weights from {active_path.name}: {e}")
        st.stop()

    model.to(DEVICE)
    model.eval()
    return model, active_path.name

# Preprocessing pipeline
preprocess_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

def predict_leaf(model, pil_img: Image.Image, classes: list, top_k: int = 5):
    """Executes EfficientNet-B0 inference and returns top-k predictions with latencies."""
    start_time = time.perf_counter()
    rgb_img = pil_img.convert("RGB")
    tensor = preprocess_transform(rgb_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(classes)))
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    results = []
    for p, idx in zip(top_probs, top_indices):
        results.append((classes[idx.item()], float(p.item()) * 100.0))

    return results, elapsed_ms

# ---------------------------------------------------------------------------------
# CLASS NAME NORMALIZATION & HELPERS
# ---------------------------------------------------------------------------------
TYPO_MAPPINGS = {
    "gauva": "guava",
    "coriender": "coriander",
    "bhrami": "brahmi",
    "astma_weed": "asthma_weed",
    "amruthaballi": "amrutha_balli",
    "citron_lime_herelikai": "citron_lime_herale",
    "images_to_predict": "unclassified_foliage",
}

def clean_class_name(raw_name: str) -> dict:
    """Normalizes raw dataset label names, strips internal tokens, and identifies disease status."""
    is_diseased = "diseased" in raw_name.lower()
    norm = raw_name.lower()

    # Apply typo corrections
    for k, v in TYPO_MAPPINGS.items():
        if k in norm:
            norm = norm.replace(k, v)

    # Remove disease marker
    norm = norm.replace("_diseased", "").replace("-diseased", "")

    # Remove internal dataset tokens like _p0a, _p11b, _p2, etc.
    tokens = [t for t in norm.replace("-", "_").split("_") if t]
    cleaned_tokens = []
    for token in tokens:
        # Check if token is internal dataset artifact (e.g., p0, p1a, p11b, p2)
        if len(token) <= 4 and token.startswith("p") and any(c.isdigit() for c in token):
            continue
        cleaned_tokens.append(token)

    base_species = " ".join(cleaned_tokens).title()
    if not base_species:
        base_species = raw_name.replace("_", " ").title()

    display_name = f"{base_species} (Diseased)" if is_diseased else f"{base_species}"

    # Determine species key for knowledge base lookups
    species_key = "_".join(cleaned_tokens).lower()

    return {
        "raw": raw_name,
        "clean_species": base_species,
        "display_name": display_name,
        "is_diseased": is_diseased,
        "species_key": species_key,
    }

# ---------------------------------------------------------------------------------
# INITIALIZE MODEL & LABELS
# ---------------------------------------------------------------------------------
raw_classes = load_class_names(CLASS_NAMES_PATH)
model, loaded_model_name = load_model(MODEL_CANDIDATES, len(raw_classes))

parsed_classes = [clean_class_name(c) for c in raw_classes]
diseased_classes = [c for c in parsed_classes if c["is_diseased"]]
healthy_classes = [c for c in parsed_classes if not c["is_diseased"]]
unique_species = sorted(list({c["clean_species"] for c in parsed_classes}))

# ---------------------------------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-wrapper">
        <h1 class="hero-title">{APP_NAME}</h1>
        <p class="hero-tagline">
            Next-generation botanical artificial intelligence. Upload or snap a leaf photo
            to identify species taxonomy and detect fungal, bacterial, or blight infections with clinical accuracy.
        </p>
        <div class="hero-badges-row">
            <span class="hero-badge">🧠 EfficientNet-B0 Engine</span>
            <span class="hero-badge">🍃 84 Recognized Classes</span>
            <span class="hero-badge">🔬 Pathology Detection Active</span>
            <span class="hero-badge">⚡ Weights: {loaded_model_name}</span>
            <span class="hero-badge">🛡️ Offline CPU Capable</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?w=600&auto=format&fit=crop&q=80",
        caption="Leaf-ID Botanical Intelligence",
        use_container_width=True,
    )

    st.markdown("### 📊 Engine Status")
    c_m1, c_m2 = st.columns(2)
    c_m1.metric("Catalog Classes", len(raw_classes))
    c_m2.metric("Disease Models", len(diseased_classes))

    st.markdown("---")
    st.markdown("### 📸 Photography Guidelines")
    st.markdown(
        """
        - **Subject Focus:** Center a **single leaf** flat against a plain, contrasting backdrop.
        - **Lighting:** Use bright, diffused natural light (avoid hard flash reflections or harsh shadows).
        - **Resolution:** Ensure vein structure and any discolored lesions are clearly in focus.
        - **Orientation:** Fill at least 60% of the camera frame with the lamina.
        """
    )

    st.markdown("---")
    st.markdown("### 📜 Session Analytics")
    st.write(f"Scans this session: **{len(st.session_state.scan_history)}**")
    if st.session_state.scan_history:
        if st.button("🧹 Clear Scan History", use_container_width=True):
            st.session_state.scan_history = []
            st.rerun()

    st.caption("Powered by PyTorch & EfficientNet • MIT License")

# ---------------------------------------------------------------------------------
# NAVIGATION TABS
# ---------------------------------------------------------------------------------
tab_diagnose, tab_directory, tab_model = st.tabs([
    "🌿 Leaf Diagnosis Hub",
    "📚 Botanical Directory (84 Classes)",
    "🔬 Model Architecture & Specs",
])

# =================================================================================
# TAB 1: LEAF DIAGNOSIS HUB
# =================================================================================
with tab_diagnose:
    st.markdown("### 1. Select Input Source")

    input_mode = st.radio(
        "Choose how you want to provide your leaf image:",
        ["📁 Upload Image File", "📷 Live Camera Capture", "🧪 Quick-Test Demo Samples"],
        horizontal=True,
        label_visibility="collapsed",
    )

    active_image = None
    image_source_label = ""

    # Mode 1: File Uploader
    if input_mode == "📁 Upload Image File":
        uploaded_file = st.file_uploader(
            "Upload leaf photograph (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Drag and drop or browse files. Images are processed locally on CPU.",
            key="file_uploader",
        )
        if uploaded_file is not None:
            try:
                active_image = Image.open(uploaded_file)
                image_source_label = f"Uploaded File: {uploaded_file.name}"
            except Exception as e:
                st.error(f"Failed to read image file: {e}")

    # Mode 2: Camera Capture
    elif input_mode == "📷 Live Camera Capture":
        st.info("💡 Grant camera permission in your browser if prompted. Center the leaf in the viewfinder.")
        cam_pic = st.camera_input("Take a photo of a leaf", key="leaf_camera")
        if cam_pic is not None:
            try:
                active_image = Image.open(cam_pic)
                image_source_label = "Live Camera Capture"
            except Exception as e:
                st.error(f"Failed to process camera capture: {e}")

    # Mode 3: Built-in Sample Gallery
    else:
        st.markdown("**Click any pre-loaded sample below to test the classifier instantly:**")
        sample_cols = st.columns(4)

        samples_meta = [
            ("sample_mango.jpg", "Healthy Mango Leaf", "🥭 Mango (Healthy)"),
            ("sample_diseased_lemon.jpg", "Diseased Lemon Leaf", "🍋 Lemon (Canker)"),
            ("sample_aloevera.jpg", "Healthy Aloe Vera", "🪴 Aloe Vera (Healthy)"),
            ("sample_diseased_tomato.jpg", "Diseased Tomato Leaf", "🍅 Tomato (Blight)"),
        ]

        for i, (fname, title, desc) in enumerate(samples_meta):
            fpath = SAMPLES_DIR / fname
            with sample_cols[i]:
                if fpath.exists():
                    img_thumb = Image.open(fpath)
                    st.image(img_thumb, caption=desc, use_container_width=True)
                    if st.button(f"Load {title}", key=f"btn_sample_{i}", use_container_width=True):
                        st.session_state.selected_sample = fpath
                        st.rerun()
                else:
                    st.caption(f"{title} (File missing)")

        if st.session_state.selected_sample and Path(st.session_state.selected_sample).exists():
            active_image = Image.open(st.session_state.selected_sample)
            image_source_label = f"Demo Sample: {Path(st.session_state.selected_sample).name}"

    # -----------------------------------------------------------------------------
    # DIAGNOSIS PROCESSING & RESULTS
    # -----------------------------------------------------------------------------
    if active_image is not None:
        st.markdown("---")
        col_img, col_diag = st.columns([1, 1.25], gap="large")

        with col_img:
            st.markdown(f"#### 🔍 Specimen Inspection")
            st.image(active_image, caption=image_source_label, use_container_width=True)

            # Metadata details
            w, h = active_image.size
            st.caption(f"📏 Dimensions: **{w} × {h} px** • Format: **{active_image.format or 'RGB'}** • Mode: **{active_image.mode}**")

        with col_diag:
            st.markdown("#### 📋 Diagnostic Report")

            with st.spinner("Analyzing botanical morphology and pathology signatures..."):
                predictions, latency_ms = predict_leaf(model, active_image, raw_classes, top_k=5)

            top_raw, top_conf = predictions[0]
            top_parsed = clean_class_name(top_raw)
            is_diseased = top_parsed["is_diseased"]

            # Session history update
            history_item = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "species": top_parsed["clean_species"],
                "diseased": is_diseased,
                "confidence": top_conf,
            }
            if not st.session_state.scan_history or st.session_state.scan_history[-1]["timestamp"] != history_item["timestamp"]:
                st.session_state.scan_history.append(history_item)

            # Dynamic Verdict Banner
            if top_conf < 45.0:
                banner_class = "verdict-inconclusive"
                icon = "⚠️"
                verdict_title = f"{icon} Inconclusive Match ({top_parsed['clean_species']})"
                verdict_subtitle = "Confidence is low. Please reposition the leaf with flat illumination and retake."
            elif is_diseased:
                banner_class = "verdict-diseased"
                icon = "🚨"
                verdict_title = f"{icon} Pathology Detected: {top_parsed['clean_species']}"
                verdict_subtitle = "Signs of plant disease or bacterial/fungal lesion were detected."
            else:
                banner_class = "verdict-healthy"
                icon = "🌿"
                verdict_title = f"{icon} Healthy Specimen: {top_parsed['clean_species']}"
                verdict_subtitle = "Foliage shows uniform pigmentation and healthy vigor."

            st.markdown(
                f"""
                <div class="verdict-banner {banner_class}">
                    <div>
                        <div style="font-size:1.25rem; font-weight:800;">{verdict_title}</div>
                        <div style="font-size:0.88rem; opacity:0.9; margin-top:3px;">{verdict_subtitle}</div>
                    </div>
                    <div style="font-size:1.6rem; font-weight:800;">{top_conf:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence Level Breakdown
            c_gauge1, c_gauge2 = st.columns(2)
            conf_tier = "High Confidence" if top_conf >= 80 else ("Moderate Confidence" if top_conf >= 50 else "Low / Uncertain")
            c_gauge1.metric("Match Confidence", f"{top_conf:.2f}%", delta=conf_tier)
            c_gauge2.metric("Inference Latency", f"{latency_ms:.1f} ms", delta="CPU Engine", delta_color="off")

            st.markdown("##### 🏆 Top Candidate Predictions")
            colors = ["#10b981", "#34d399", "#6ee7b7", "#a7f3d0", "#d1fae5"]
            if is_diseased:
                colors[0] = "#ef4444"

            for rank, (cand_raw, cand_conf) in enumerate(predictions[:4], start=1):
                cand_info = clean_class_name(cand_raw)
                bar_color = "#ef4444" if cand_info["is_diseased"] else "#10b981"

                st.markdown(
                    f"""
                    <div class="pred-bar-container">
                        <div class="pred-header">
                            <span>#{rank} <b>{cand_info['display_name']}</b></span>
                            <span>{cand_conf:.1f}%</span>
                        </div>
                        <div class="pred-track">
                            <div class="pred-fill" style="width:{cand_conf:.1f}%; background-color:{bar_color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # -----------------------------------------------------------------------------
        # BOTANICAL ADVISORY & TREATMENT EXPANDER
        # -----------------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 💊 Botanical Advisory & Care Plan")

        species_key = top_parsed["species_key"]
        advice_info = BOTANICAL_ADVICE.get(species_key, None)

        # Fallback partial matching if direct key not matched
        if not advice_info:
            for k in BOTANICAL_ADVICE:
                if k in species_key or species_key in k:
                    advice_info = BOTANICAL_ADVICE[k]
                    break

        adv_col1, adv_col2 = st.columns([1.5, 1], gap="medium")

        with adv_col1:
            if is_diseased:
                if advice_info and "diseased" in advice_info:
                    d_data = advice_info["diseased"]
                    st.error(f"**Identified Condition:** {d_data['condition']}")
                    st.markdown(f"**Observed Symptoms:** {d_data['symptoms']}")
                    st.markdown(f"**Probable Cause:** {d_data['cause']}")

                    st.markdown("#### 🛡️ Recommended Intervention Steps")
                    for step in d_data["treatment"]:
                        st.markdown(f"- ✅ {step}")
                else:
                    st.warning(
                        f"Specific pathology monograph for **{top_parsed['clean_species']}** is not in our direct database. "
                        "General intervention: Isolate infected plants, prune damaged foliage with disinfected shears, "
                        "and apply a broad-spectrum copper or bio-fungicidal spray."
                    )
            else:
                if advice_info and "healthy" in advice_info:
                    st.success(f"**Foliage Vigor:** Excellent. No immediate pathogen intervention required.")
                    st.markdown(f"**Care & Cultivation:** {advice_info['healthy']['care']}")
                else:
                    st.success(
                        f"**Foliage Vigor:** Specimen matches healthy **{top_parsed['clean_species']}** foliage. "
                        "Continue optimal irrigation, nutrient monitoring, and inspection for pests."
                    )

        with adv_col2:
            st.markdown("#### 📥 Diagnostic Certificate")
            report_text = f"""====================================================
LEAF ID PRO — BOTANICAL DIAGNOSTIC REPORT
Date & Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
====================================================
SPECIMEN ANALYSIS:
Primary Species: {top_parsed['clean_species']}
Pathology Status: {'DISEASED' if is_diseased else 'HEALTHY'}
Model Verdict: {top_parsed['display_name']}
Confidence Score: {top_conf:.2f}%
Model Architecture: EfficientNet-B0 (PyTorch)
Inference Device: CPU ({latency_ms:.1f} ms)

TOP PREDICTIONS:
"""
            for i, (c_raw, c_conf) in enumerate(predictions, 1):
                c_clean = clean_class_name(c_raw)
                report_text += f"{i}. {c_clean['display_name']} — {c_conf:.2f}%\n"

            report_text += f"""
RECOMMENDED ACTION:
{('Consult local plant pathology extension or apply targeted fungicide/bactericide.' if is_diseased else 'Maintain standard watering, sunlight, and soil maintenance.')}
====================================================
"""
            st.download_button(
                label="📄 Download Diagnostic Report (.txt)",
                data=report_text,
                file_name=f"leafid_report_{top_parsed['clean_species'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

            with st.expander("🔬 View Raw Prediction Distribution"):
                df_preds = pd.DataFrame([
                    {"Class": clean_class_name(r)["display_name"], "Raw ID": r, "Confidence (%)": round(c, 2)}
                    for r, c in predictions
                ])
                st.dataframe(df_preds, hide_index=True, use_container_width=True)

    else:
        # Empty state guidance
        st.info("👆 Please upload an image, capture via camera, or select a demo sample to begin diagnosis.")

        st.markdown(
            """
            <div class="bio-card">
                <h4>✨ Features of LEAF ID Pro</h4>
                <ul>
                    <li><b>Dual-Action Detection:</b> Identifies species taxonomy while simultaneously diagnosing foliar pathology.</li>
                    <li><b>Broad Botanical Vocabulary:</b> Covers 84 classes of commercial crops, fruit trees, medicinal herbs, and ornamental plants.</li>
                    <li><b>Zero Cloud Dependency:</b> Complete PyTorch deep learning pipeline executes locally on CPU with zero privacy leakage.</li>
                    <li><b>Actionable Pathology Guide:</b> Provides immediate cultural and organic treatment remedies for identified diseases.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =================================================================================
# TAB 2: BOTANICAL SPECIES DIRECTORY
# =================================================================================
with tab_directory:
    st.markdown("### 📚 Supported Botanical Taxonomy")
    st.write(
        f"LEAF ID Pro is trained across **{len(raw_classes)} distinct classes**, spanning common agricultural staples, "
        "tropical fruit trees, ayurvedic & traditional herbs, and garden ornamentals."
    )

    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Total Vocabulary Classes", len(raw_classes))
    stat2.metric("Diseased State Categories", len(diseased_classes))
    stat3.metric("Species-Only / Healthy Categories", len(healthy_classes))

    st.markdown("---")

    col_search, col_filter = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("🔍 Search species or disease", placeholder="e.g. mango, tomato, diseased, basil, lemon...")
    with col_filter:
        filter_type = st.selectbox("Category Filter", ["All Classes", "Healthy / Species Only", "Disease-Monitored Species"])

    # Filter logic
    filtered_list = parsed_classes
    if filter_type == "Healthy / Species Only":
        filtered_list = [c for c in filtered_list if not c["is_diseased"]]
    elif filter_type == "Disease-Monitored Species":
        filtered_list = [c for c in filtered_list if c["is_diseased"]]

    if search_query:
        sq = search_query.lower()
        filtered_list = [c for c in filtered_list if sq in c["display_name"].lower() or sq in c["raw"].lower()]

    st.caption(f"Showing **{len(filtered_list)}** matching classes:")

    grid_cols = st.columns(3)
    for idx, item in enumerate(filtered_list):
        with grid_cols[idx % 3]:
            tag_class = "plant-tag-diseased" if item["is_diseased"] else "plant-tag"
            icon = "🚨" if item["is_diseased"] else "🌱"
            st.markdown(
                f"""
                <div class="{tag_class}">
                    {icon} <b>{item['display_name']}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =================================================================================
# TAB 3: MODEL ARCHITECTURE & SPECS
# =================================================================================
with tab_model:
    st.markdown("### 🔬 Neural Network Architecture & Technical Pipeline")

    arch_col1, arch_col2 = st.columns([1.2, 1], gap="large")

    with arch_col1:
        st.markdown(
            """
            #### 🧬 EfficientNet-B0 Backbone
            LEAF ID Pro leverages the **EfficientNet-B0** convolutional neural network, fine-tuned
            via transfer learning for high-precision botanical feature extraction.

            - **Compound Scaling:** Uniformly balances depth ($d=1.0$), width ($w=1.0$), and resolution ($r=1.0$) using fixed scaling coefficients.
            - **Mobile Inverted Bottleneck (MBConv):** Employs depthwise separable convolutions with Squeeze-and-Excitation (SE) attention blocks for minimal parameter overhead (~5.3M parameters).
            - **Custom Classification Head:** Replaces ImageNet's 1000-class linear projection with a specialized `nn.Linear(1280, 84)` output layer.
            """
        )

        st.markdown("#### 🔄 Preprocessing & Inference Flow")
        steps = [
            ("1", "Input Ingestion", "Accepts standard JPG/PNG/WEBP and converts color space to 3-channel RGB."),
            ("2", "Geometric Scaling", f"Bilinear interpolation resizes specimen lamina to {IMG_SIZE}×{IMG_SIZE} px."),
            ("3", "Standardization", f"Normalizes with ImageNet coefficients: μ={NORM_MEAN}, σ={NORM_STD}."),
            ("4", "Logit Projection", "Passes through EfficientNet-B0 and applies Softmax activation across 84 logits."),
        ]
        s_cols = st.columns(4)
        for c, (num, title, desc) in zip(s_cols, steps):
            with c:
                st.markdown(
                    f"""
                    <div class="step-box">
                        <div class="step-num">{num}</div>
                        <div style="font-weight:700; font-size:0.95rem; margin-bottom:6px;">{title}</div>
                        <div style="font-size:0.8rem; opacity:0.85;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with arch_col2:
        st.markdown("#### ⚙️ Technical Specifications")
        specs_df = pd.DataFrame({
            "Specification": [
                "Architecture",
                "Weight Checkpoint",
                "Input Dimensions",
                "Output Classes",
                "Runtime Device",
                "Framework",
                "Dataset Sources"
            ],
            "Value": [
                "EfficientNet-B0 (MBConv + SE)",
                loaded_model_name,
                f"{IMG_SIZE} × {IMG_SIZE} × 3",
                f"{len(raw_classes)} Classes",
                "CPU (Offline Capable)",
                f"PyTorch {torch.__version__}",
                "Kaggle (48 Plant Leaves / Plant Leaf / Classification)"
            ]
        })
        st.dataframe(specs_df, hide_index=True, use_container_width=True)

        st.markdown("#### 📚 Open Datasets Attributions")
        st.markdown(
            """
            - [48 Plant Leaves Datasets](https://www.kaggle.com/datasets/developerzulkarnain/48-plant-leaves-datasets)
            - [Plant Leaf Dataset](https://www.kaggle.com/datasets/mahaninghubballi/plant-leaf-dataset)
            - [Plant Leaves for Image Classification](https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification)
            """
        )

# ---------------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; opacity: 0.7; font-size: 0.85rem; padding: 10px 0;">
        🌿 <b>LEAF ID Pro</b> — Open-Source Plant Recognition & Pathology Diagnostic Tool • Built with PyTorch & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)