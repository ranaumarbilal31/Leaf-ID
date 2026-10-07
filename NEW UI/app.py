# ─────────────────────────────────────────────────────────────────────────────
# LEAF ID · app.py — field-lab UI redesign
# Stack: Streamlit + PyTorch/torchvision (EfficientNet-B0) + Pillow.
# Run:   streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

import base64
import hashlib
import html as _html
import io
import json
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

# ─────────────────────────────────────────────────────────────────────────────
# 0 · PAGE CONFIG (must be the first Streamlit call) + paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

# Support running from root or subdirectories, and check candidate weights
MODEL_PATH = ROOT / "LeafID.pt"
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "braincell_best.pt"
if not MODEL_PATH.exists() and (ROOT.parent / "LeafID.pt").exists():
    MODEL_PATH = ROOT.parent / "LeafID.pt"
if not MODEL_PATH.exists() and (ROOT.parent / "braincell_best.pt").exists():
    MODEL_PATH = ROOT.parent / "braincell_best.pt"

CLASSES_PATH = ROOT / "class_names.json"
if not CLASSES_PATH.exists() and (ROOT.parent / "class_names.json").exists():
    CLASSES_PATH = ROOT.parent / "class_names.json"

STYLE_PATH = ROOT / "style.css"
if not STYLE_PATH.exists() and (ROOT.parent / "style.css").exists():
    STYLE_PATH = ROOT.parent / "style.css"

SAMPLES_DIR = ROOT / "samples"
if not SAMPLES_DIR.exists() and (ROOT.parent / "samples").exists():
    SAMPLES_DIR = ROOT.parent / "samples"

CONFIDENCE_FLOOR = 0.45   # below this, show the neutral "uncertain read" state
MAX_PREVIEW_PX = 640      # preview thumbnail size (display only — never saved)
MAX_VIDEO_MB = 8          # hero video is base64-embedded only if small enough
REPO_URL = "https://github.com/ranaumarbilal31/Leaf-ID"
ISSUES_URL = REPO_URL + "/issues"
PRS_URL = REPO_URL + "/pulls"
FORK_URL = REPO_URL + "/fork"

_FAVICON = ASSETS / "favicon.svg"
if not _FAVICON.exists() and (ROOT.parent / "assets" / "favicon.svg").exists():
    _FAVICON = ROOT.parent / "assets" / "favicon.svg"

st.set_page_config(
    page_title="LEAF ID — AI leaf identification",
    page_icon=str(_FAVICON) if _FAVICON.exists() else "🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1 · CLASS REGISTRY & BOTANICAL ADVISORY
# ─────────────────────────────────────────────────────────────────────────────
def load_class_names(path):
    """Accepts a JSON list or a {"0": label, ...} dict; returns [str, ...]."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        try:
            data = [data[k] for k in sorted(data, key=lambda x: int(x))]
        except Exception:
            return []
    return [str(n).strip() for n in data if str(n).strip()]

CLASS_NAMES = load_class_names(CLASSES_PATH)
N_CLASSES = len(CLASS_NAMES) if CLASS_NAMES else 84


def parse_label(raw):
    """Parses raw class names, normalizes typos, strips dataset artifacts, identifies disease."""
    s = str(raw).strip()
    is_diseased = "diseased" in s.lower()

    norm = s.lower().replace("_diseased", "").replace("-diseased", "")
    for typo, fix in [
        ("gauva", "guava"),
        ("coriender", "coriander"),
        ("bhrami", "brahmi"),
        ("astma_weed", "asthma_weed"),
        ("amruthaballi", "amrutha_balli"),
        ("citron_lime_herelikai", "citron_lime"),
        ("images_to_predict", "unclassified_foliage"),
    ]:
        norm = norm.replace(typo, fix)

    tokens = [t for t in norm.replace("-", "_").split("_") if t]
    clean_tokens = [t for t in tokens if not (len(t) <= 4 and t.startswith("p") and any(c.isdigit() for c in t))]

    species = " ".join(clean_tokens).title() if clean_tokens else s.title()
    cond = "Diseased" if is_diseased else "Healthy"
    healthy = not is_diseased
    return species, cond, healthy


SPECIES_STATES = {}  # species → set of conditions
for _n in CLASS_NAMES:
    _sp, _cd, _h = parse_label(_n)
    SPECIES_STATES.setdefault(_sp, set()).add(_cd)

DISEASE_SPECIES = {sp for sp, cds in SPECIES_STATES.items() if "Diseased" in cds}
N_DISEASE = len(DISEASE_SPECIES) if DISEASE_SPECIES else 11

# Botanical advisory for key pathology classes
BOTANICAL_CARE = {
    "Lemon": {
        "condition": "Citrus Canker / Bacterial Blight",
        "symptoms": "Raised corky lesions surrounded by yellow chlorotic halos on foliage and twigs.",
        "treatment": "Prune severely affected twigs; apply copper-based bactericide/fungicide spray; avoid overhead watering.",
    },
    "Mango": {
        "condition": "Anthracnose / Leaf Blight",
        "symptoms": "Dark brown, angular to irregular necrotic spots that coalesce into large blights.",
        "treatment": "Prune canopy to improve air circulation; apply neem oil or copper oxychloride fungicide every 14 days.",
    },
    "Tomato": {
        "condition": "Early Blight / Septoria Leaf Spot",
        "symptoms": "Concentric target-like brown spots with yellow halos, starting on lower leaves.",
        "treatment": "Remove lower foliage within 12 inches of soil; mulch base heavily; spray organic copper or bio-fungicide.",
    },
    "Guava": {
        "condition": "Guava Wilt / Anthracnose",
        "symptoms": "Curling, yellowing leaves with rusty-brown spots leading to premature defoliation.",
        "treatment": "Drench root zone with Trichoderma viride; ensure soil drainage; apply balanced micronutrients.",
    },
    "Pomegranate": {
        "condition": "Bacterial Blight / Cercospora Spot",
        "symptoms": "Water-soaked dark brown spots turning black with chlorotic margins.",
        "treatment": "Prune during dry spells with sterilized shears; apply copper hydroxide combined with agricultural bactericide.",
    },
    "Jamun": {
        "condition": "Leaf Spot / Anthracnose",
        "symptoms": "Circular reddish-brown specks expanding into shot-hole lesions.",
        "treatment": "Apply 1% Bordeaux mixture; rake and destroy fallen leaf litter beneath the tree canopy.",
    },
    "Jatropha": {
        "condition": "Powdery Mildew / Rust",
        "symptoms": "White powdery fungal coating on leaf upper surfaces causing distortion and leaf drop.",
        "treatment": "Apply potassium bicarbonate or wettable sulfur spray; thin dense branches for interior airflow.",
    },
    "Chinar": {
        "condition": "Sycamore Anthracnose",
        "symptoms": "Brown necrotic lesions following leaf veins; sudden blight of young spring foliage.",
        "treatment": "Rake and destroy fallen leaves in autumn; apply systemic fungicide for heritage specimens.",
    },
    "Alstonia Scholaris": {
        "condition": "Leaf Gall / Insect Blister",
        "symptoms": "Raised blister-like gall formations on the upper leaf surface caused by psyllid midges.",
        "treatment": "Prune heavily galled leaves before insects emerge; apply 5% neem seed kernel extract spray.",
    },
    "Pongamia Pinnata": {
        "condition": "Tar Spot / Gall Mite Blight",
        "symptoms": "Black tar-like raised fungal spots or blistered galls across the lamina.",
        "treatment": "Apply wettable sulfur or horticultural oil during early spring; clear fallen leaf debris.",
    },
    "Arjun": {
        "condition": "Leaf Rust / Gall Formation",
        "symptoms": "Orange-yellowish pustules on leaf underside or irregular swelling on veins.",
        "treatment": "Prune isolated affected leaves; spray copper oxychloride or bio-fungicide if spreading.",
    },
}

CATEGORY_OF = {k.lower(): v for k, v in {
    # Fruit trees
    "Apple": "fruit", "Grape": "fruit", "Peach": "fruit", "Cherry": "fruit",
    "Orange": "fruit", "Strawberry": "fruit", "Blueberry": "fruit", "Raspberry": "fruit",
    "Lemon": "fruit", "Mango": "fruit", "Guava": "fruit", "Pomegranate": "fruit",
    "Jamun": "fruit", "Jackfruit": "fruit", "Amla": "fruit", "Bael": "fruit",
    # Garden vegetables
    "Tomato": "vegetable", "Potato": "vegetable", "Pepper": "vegetable",
    "Corn": "vegetable", "Squash": "vegetable", "Beans": "vegetable",
    "Malabar Spinach": "vegetable", "Drumstick": "vegetable", "Chilly": "vegetable",
    # Medicinal
    "Aloevera": "medicinal", "Basil": "medicinal", "Mint": "medicinal",
    "Neem": "medicinal", "Ginger": "medicinal", "Turmeric": "medicinal",
    "Eucalyptus": "medicinal", "Coriander": "medicinal", "Amrutha Balli": "medicinal",
    "Arjun": "medicinal", "Ashoka": "medicinal", "Brahmi": "medicinal",
    "Asthma Weed": "medicinal", "Balloon Vine": "medicinal", "Betel": "medicinal",
    "Nelavembu": "medicinal", "Henna": "medicinal", "Insulin": "medicinal",
    # Ornamental
    "Rose": "ornamental", "Hibiscus": "ornamental", "Marigold": "ornamental",
    "Jasmine": "ornamental", "Chinar": "ornamental", "Alstonia Scholaris": "ornamental",
    "Pongamia Pinnata": "ornamental", "Jatropha": "ornamental", "Bamboo": "ornamental",
    "Caricature": "ornamental", "Catharanthus": "ornamental", "Castor": "ornamental",
}.items()}

CATEGORY_LABELS = {
    "fruit": "Fruit Trees",
    "vegetable": "Vegetables",
    "medicinal": "Medicinal",
    "ornamental": "Ornamental",
    "other": "Other & Field",
}

def category_for(species):
    s = species.strip().lower()
    if s in CATEGORY_OF:
        return CATEGORY_OF[s]
    first = s.split(" ")[0].split("(")[0].strip()
    return CATEGORY_OF.get(first, "other")

# ─────────────────────────────────────────────────────────────────────────────
# 2 · MODEL / INFERENCE CORE
# ─────────────────────────────────────────────────────────────────────────────
def _torch_load(path):
    """Checkpoint loader that survives torch version differences."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        pass
    except Exception:
        pass
    return torch.load(str(path), map_location="cpu")


@st.cache_resource(show_spinner=False)
def load_model():
    """EfficientNet-B0 fine-tuned checkpoint — loaded once, cached forever."""
    try:
        net = models.efficientnet_b0(weights=None)
    except TypeError:
        net = models.efficientnet_b0(pretrained=False)

    state = _torch_load(MODEL_PATH)
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    elif isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    net.load_state_dict(state, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return net.to(device).eval()


EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_data(show_spinner=False)
def run_inference(image_bytes):
    """Single-leaf forward pass → [(class_label, probability), …] ranked."""
    model = load_model()
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    tensor = EVAL_TRANSFORMS(img).unsqueeze(0)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    ranked = sorted(zip(CLASS_NAMES, probs.tolist()), key=lambda t: t[1], reverse=True)
    return ranked

# ─────────────────────────────────────────────────────────────────────────────
# 3 · UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def esc(s):
    return _html.escape(str(s), quote=True)

def b64_data(path, mime):
    try:
        return "data:%s;base64,%s" % (mime, base64.b64encode(path.read_bytes()).decode("ascii"))
    except Exception:
        return None

def preview_data_uri(image_bytes):
    """Downscaled in-memory preview (no disk writes)."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((MAX_PREVIEW_PX, MAX_PREVIEW_PX))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return (
            "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            img.size,
        )
    except Exception:
        return None, (0, 0)

def safe_slug(s):
    return "".join(c if c.isalnum() else "-" for c in s.lower()).strip("-")

def leaf_placeholder_uri(seed):
    """Deterministic archival-plate leaf glyph (SVG data-URI)."""
    h = int(hashlib.md5(seed.encode("utf-8")).hexdigest()[:6], 16)
    hue = 88 + (h % 64)
    rot = (h >> 3) % 30 - 15
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 160'>"
        "<g transform='rotate(%d 60 80)' fill='none' stroke='hsl(%d,70%%,62%%)'>"
        "<path d='M60 14 C 98 46 106 94 60 146 C 14 94 22 46 60 14 Z' stroke-width='2.4' opacity='.95'/>"
        "<path d='M60 26 L60 138' stroke-width='1.5' opacity='.8'/>"
        "<path d='M60 50 C 74 54 86 62 92 72 M60 50 C 46 54 34 62 28 72 "
        "M60 78 C 72 82 82 88 88 96 M60 78 C 48 82 38 88 32 96 "
        "M60 106 C 70 110 78 114 83 121 M60 106 C 50 110 42 114 37 121' stroke-width='1' opacity='.5'/>"
        "</g></svg>" % (rot, hue)
    )
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg, safe="'/=:,.-")

FALLBACK_CSS = """
html{scroll-behavior:smooth}
body{background:#07100B;color:#E9F2E4}
.stApp{background:transparent;color:#E9F2E4}
header[data-testid="stHeader"],footer[data-testid="stFooter"],#MainMenu,
[data-testid="stSidebar"],[data-testid="stToolbar"],#stDecoration{display:none!important}
"""

def inject_css():
    """Load style.css via plain file read; keep a minimal dark safety net."""
    css = FALLBACK_CSS
    try:
        css += STYLE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass
    st.markdown("<style>%s</style>" % css, unsafe_allow_html=True)

def section_header(idx, eyebrow, title, sub=""):
    sub_html = '<p class="sec-sub">%s</p>' % sub if sub else ""
    return (
        '<div class="sec-head reveal" style="--d:.05s">'
        '<div class="sec-eyebrow"><span class="tick"></span>%s</div>'
        '<h2 class="sec-title">%s</h2>%s</div>' % (esc(eyebrow), title, sub_html)
    )

def empty_state(note=None):
    note_html = (
        '<p class="se-note warn">%s</p>' % esc(note)
        if note
        else '<p class="se-note">Drop a JPG or PNG of a single leaf to begin a scan.</p>'
    )
    return (
        '<div class="spec-empty reveal" style="--d:.1s" role="status">'
        '<div class="crosshair" aria-hidden="true"></div>'
        '<h4>NO SPECIMEN LOADED</h4>%s</div>' % note_html
    )

def progress_step(p, text=None):
    try:
        return st.progress(p, text=text) if text else st.progress(p)
    except TypeError:
        return st.progress(p)

def specimen_card(filename, uri, size, ranked, seq, animate):
    top_label, top_p = ranked[0]
    sp, cond, healthy = parse_label(top_label)
    pct = max(0.0, min(100.0, top_p * 100.0))
    uncertain = top_p < CONFIDENCE_FLOOR
    capable = sp in DISEASE_SPECIES

    if uncertain:
        flag_cls = "uncertain"
        flag = "UNCERTAIN READ — TRY A CLEARER, EVENLY-LIT PHOTO OF A SINGLE LEAF"
        eyebrow = "SCAN %02d · LOW-CONFIDENCE READ" % seq
    else:
        eyebrow = "SCAN %02d · READOUT COMPLETE" % seq
        if healthy and capable:
            flag_cls, flag = "healthy", "HEALTHY — NO DISEASE SIGNAL DETECTED"
        elif healthy:
            flag_cls, flag = "na", "SPECIES ONLY — NO DISEASE CHANNEL FOR THIS CLASS"
        else:
            flag_cls, flag = "disease", "DISEASE SIGNAL — %s" % cond.upper()

    anim_cls = " animate" if animate else ""
    scan_html = '<div class="spec-scan" aria-hidden="true"></div>' if animate else ""
    tag_txt = "SCANNING" if animate else "LIVE SCAN"
    meta = "%s · %d×%d PX · %d CLASSES SCORED" % (filename, size[0], size[1], len(ranked))
    name_html = "%s <span class='sep'>·</span> <span class='cond'>%s</span>" % (esc(sp), esc(cond))

    rows = []
    for i, (lab, p) in enumerate(ranked[1:4], start=2):
        lsp, lcd, _ = parse_label(lab)
        rows.append(
            '<div class="alt-row"><span class="rk">%02d</span>'
            '<span class="nm">%s — %s</span>'
            '<span class="alt-bar"><i style="--w:%.1f%%"></i></span>'
            '<span class="alt-pct">%.1f%%</span></div>'
            % (i, esc(lsp), esc(lcd), p * 100, p * 100)
        )
    alt_block = "".join(rows) or '<div class="alt-row"><span class="nm">No alternate reads.</span></div>'

    return """
<div class="specimen%(anim)s%(unc)s reveal" style="--d:.05s" id="leafid-result">
  <div class="spec-frame">
    <img class="spec-photo" src="%(uri)s" alt="Uploaded leaf specimen photograph">
    %(scan)s
    <span class="spec-tag tag-tl"><i class="dot" aria-hidden="true"></i>%(tag)s</span>
    <span class="spec-tag tag-br">%(meta)s</span>
  </div>
  <div class="spec-meta">
    <div class="spec-eyebrow">%(eyebrow)s</div>
    <h3 class="spec-name">%(name)s</h3>
    <div class="conf-head">
      <span class="conf-label">TOP-1 CONFIDENCE</span>
      <span class="conf-num" id="leafid-conf-num" data-target="%(pct).1f">%(pct).1f%%</span>
    </div>
    <div class="conf-bar"><i style="--w:%(pct).1f%%"></i></div>
    <div class="read-flag %(flagcls)s"><span class="dot" aria-hidden="true"></span>%(flag)s</div>
    <div class="alt-reads"><div class="alt-title">ALTERNATE READS</div>%(alts)s</div>
  </div>
</div>""" % {
        "anim": anim_cls,
        "unc": " is-uncertain" if uncertain else "",
        "uri": uri,
        "scan": scan_html,
        "tag": tag_txt,
        "meta": esc(meta),
        "eyebrow": esc(eyebrow),
        "name": name_html,
        "pct": pct,
        "flagcls": flag_cls,
        "flag": esc(flag),
        "alts": alt_block,
    }


def class_grid(items):
    cards = []
    for i, raw in enumerate(sorted(items), start=1):
        sp, cd, _ = parse_label(raw)
        cat = category_for(sp)
        capable = sp in DISEASE_SPECIES
        img_path = ASSETS / "classes" / (safe_slug(raw) + ".jpg")
        uri = b64_data(img_path, "image/jpeg") or leaf_placeholder_uri(raw)
        alt = "Reference leaf illustration for %s — %s" % (sp, cd)
        if capable:
            badge = '<span class="d-badge">DISEASE READOUT · %d STATES</span>' % len(SPECIES_STATES[sp])
            note = "Class set carries healthy + disease labels; a diseased scan raises an amber flag."
        else:
            badge = '<span class="h-badge">SPECIES ONLY</span>'
            note = "No disease labels for this species — reports species and confidence."
        cards.append(
            '<div class="lc" tabindex="0" role="group" aria-label="%s — details on flip side">'
            '<div class="lc-inner">'
            '<div class="lc-face lc-front"><span class="lc-idx">%03d</span>'
            '<div class="lc-media"><img class="lc-img" src="%s" alt="%s" loading="lazy"></div>'
            '<div class="nm">%s</div><div class="sp">%s</div></div>'
            '<div class="lc-face lc-back"><span class="cat">%s</span>%s<p>%s</p>'
            '<span class="lc-id">%s</span></div>'
            '</div></div>'
            % (
                esc(raw),
                i,
                uri,
                esc(alt),
                esc(sp),
                esc(cd.upper()),
                esc(CATEGORY_LABELS[cat].upper()),
                badge,
                esc(note),
                esc(raw),
            )
        )
    return '<div class="class-grid">' + "".join(cards) + "</div>"

# ─────────────────────────────────────────────────────────────────────────────
# 4 · STATIC COPY & TRANSPARENCY
# ─────────────────────────────────────────────────────────────────────────────
KAGGLE_DATASETS = [
    ("48 Plant Leaves Datasets", "https://www.kaggle.com/datasets/developerzulkarnain/48-plant-leaves-datasets",
     "Real-world crop and medicinal leaf images across healthy and diseased conditions."),
    ("Plant Leaf Dataset", "https://www.kaggle.com/datasets/mahaninghubballi/plant-leaf-dataset",
     "Labelled medicinal and field crop leaf imagery for class expansion."),
    ("Plant Leaves for Image Classification", "https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification",
     "Curated multi-class botanical foliage datasets for fine-tuning and evaluation."),
]

STEPS = [
    ("01", "Capture", "Photograph one leaf flat against a plain, contrasting surface. Fill the frame and keep the blade in focus."),
    ("02", "Ingest", "Upload a JPG/PNG, capture live via camera, or select a demo specimen. Image processing executes locally in-memory."),
    ("03", "Classify", "EfficientNet-B0 compares vein and morphology features against indexed botanical classes."),
    ("04", "Diagnostic readout", "Species identified first, disease symptoms and organic treatment guidance provided where supported."),
]
TIPS = ["SINGLE LEAF", "FLAT & STILL", "EVEN LIGHT", "PLAIN BACKGROUND", "FILL THE FRAME"]

PIPELINE = [
    ("01", "Data merge", "Three public Kaggle leaf datasets merged, de-duplicated and indexed into 84 classes."),
    ("02", "Augmentation", "Flips, rotations, crops and color jitter widen field robustness before training."),
    ("03", "Fine-tune", "EfficientNet-B0 backbone, ImageNet-pretrained, adapted with an 84-class classification head."),
    ("04", "Validation", "Per-class validation audited against labelled disease states on CPU inference."),
]
TECH = ["PYTHON 3.9+", "PYTORCH", "TORCHVISION", "EFFICIENTNET-B0", "STREAMLIT", "PILLOW", "NUMPY", "PANDAS"]

LIMITATIONS = [
    "One leaf per scan — multi-leaf or whole-plant photographs fall outside the training distribution.",
    "Disease readout exists for %d classes; other botanical species report species identity and confidence." % N_DISEASE,
    "Field lighting matters: direct sun glare, motion blur, and cluttered soil depress model confidence.",
    "The model classifies within its %d indexed classes. Unfamiliar species are mapped to nearest visual matches.",
    "Not a substitute for an agronomist or plant pathologist. Treat readouts as an AI diagnostic screen.",
]

AUTHORS = [
    ("Rana Umar Bilal", "@ranaumarbilal31", "https://github.com/ranaumarbilal31", "RU"),
    ("Mian Zaid", "@zaid-mian", "https://github.com/zaid-mian", "MZ"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 5 · THEME + AMBIENT BACKGROUND
# ─────────────────────────────────────────────────────────────────────────────
inject_css()
st.session_state.setdefault("scan_seq", 0)
st.session_state.setdefault("scan_digest", None)
st.session_state.setdefault("demo_sample_path", None)

def background_stack():
    video_uri = None
    for name in ("hero.mp4", "hero.webm"):
        p = ASSETS / name
        if p.exists() and p.stat().st_size <= MAX_VIDEO_MB * 1024 * 1024:
            mime = "video/mp4" if name.endswith(".mp4") else "video/webm"
            video_uri = b64_data(p, mime)
            if video_uri:
                break
    if video_uri:
        media = (
            '<video class="bg-video" id="leafid-hero-video" autoplay muted loop playsinline '
            'tabindex="-1" aria-hidden="true"><source src="%s" type="video/mp4"></video>' % video_uri
        )
        mute = (
            '<button id="leafid-mute" class="hero-mute" type="button" title="Toggle background sound" '
            'aria-pressed="false">SOUND · OFF</button>'
        )
    else:
        media = '<div class="bg-video-fallback" aria-hidden="true"></div>'
        mute = ""
    stack = (
        '<div class="bg-stack" aria-hidden="true"><div class="bg-veins"></div>%s'
        '<div class="bg-shade"></div><div class="bg-grain"></div></div>' % media
    )
    return stack, mute

BG_STACK, MUTE_BTN = background_stack()
st.markdown(BG_STACK, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6 · HERO — headline + Scanner Bench
# ─────────────────────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.markdown(
        '<div class="inline-alert err">MODEL OFFLINE — LeafID.pt was not found. '
        'Please ensure LeafID.pt exists in the application root directory.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="hero" id="leafid-hero">
  %(mute)s
  <div class="eyebrow reveal" style="--d:0s"><span class="tick"></span>OPEN-SOURCE BOTANICAL SCAN BENCH · MIT LICENSED</div>
  <h1 class="display reveal" style="--d:.08s">Put a leaf<br>under the <span class="lme">lens</span>.</h1>
  <p class="hero-sub reveal" style="--d:.16s">LEAF ID reads a single leaf photo and returns species, disease status
  where supported, and ranked confidence — a fine-tuned EfficientNet-B0 you can audit line by line.</p>
  <div class="chip-row reveal" style="--d:.24s">
    <span class="chip"><b>%(nclasses)d</b> PLANT CLASSES</span>
    <span class="chip"><b>%(ndisease)d</b> DISEASE READOUTS</span>
    <span class="chip"><b>B0</b> EFFICIENTNET BACKBONE</span>
    <span class="chip"><b>MIT</b> OPEN LICENSE</span>
  </div>
</div>"""
    % {"mute": MUTE_BTN, "nclasses": N_CLASSES, "ndisease": N_DISEASE},
    unsafe_allow_html=True,
)

left, right = st.columns([1.08, 1], gap="large")
with left:
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Classes indexed", "%02d" % N_CLASSES)
    with m2:
        st.metric("Disease readouts", "%02d" % N_DISEASE)
    with m3:
        st.metric("Input plate", "224×224")

def handle_specimen(image_bytes, filename):
    """Intake → preview → inference → light-table result card + clinical care advice."""
    if not MODEL_PATH.exists():
        st.markdown(empty_state("MODEL OFFLINE — LeafID.pt not found"), unsafe_allow_html=True)
        return
    if not image_bytes:
        st.markdown(empty_state(), unsafe_allow_html=True)
        return
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        st.markdown(
            '<div class="inline-alert err">INVALID SPECIMEN — %s is not a readable image.</div>' % esc(filename),
            unsafe_allow_html=True,
        )
        return
    if not CLASS_NAMES:
        st.markdown(empty_state("CLASS REGISTRY MISSING — class_names.json not found"), unsafe_allow_html=True)
        return

    uri, size = preview_data_uri(image_bytes)
    prog = progress_step(8, "SCANNING SPECIMEN …")
    time.sleep(0.08)
    try:
        pset_ok = True
        prog.progress(40, text="MATCHING VEIN SIGNATURES …")
    except TypeError:
        prog.progress(40)
        pset_ok = False
    try:
        ranked = run_inference(image_bytes)
    except Exception as e:
        prog.empty()
        st.markdown(
            '<div class="inline-alert err">INFERENCE FAILED: %s. Try another photo of a single leaf.</div>' % esc(e),
            unsafe_allow_html=True,
        )
        return
    if pset_ok:
        prog.progress(100, text="READOUT COMPLETE")
    else:
        prog.progress(100)
    time.sleep(0.12)
    prog.empty()

    digest = hashlib.md5(image_bytes).hexdigest()
    if st.session_state.get("scan_digest") != digest:
        st.session_state["scan_digest"] = digest
        st.session_state["scan_seq"] = st.session_state.get("scan_seq", 0) + 1
        new_scan = True
    else:
        new_scan = False

    st.markdown(
        specimen_card(filename, uri, size, ranked, st.session_state.get("scan_seq", 1), new_scan),
        unsafe_allow_html=True,
    )
    if new_scan:
        components.html(COUNTUP_JS, height=0)

    # Botanical care & pathology advisory box
    top_label, top_p = ranked[0]
    sp, cond, healthy = parse_label(top_label)
    if not healthy and sp in BOTANICAL_CARE:
        care = BOTANICAL_CARE[sp]
        st.markdown(
            f"""
            <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 18px 20px; margin-top: 14px;">
              <div style="font-family:'JetBrains Mono', monospace; font-size:0.76rem; color:#EF4444; letter-spacing:0.08em; font-weight:700;">
                🚨 PATHOLOGY ADVISORY · {esc(care['condition'].upper())}
              </div>
              <p style="font-size:0.86rem; color:#E9F2E4; margin:8px 0 6px 0;"><b>Observed Symptoms:</b> {esc(care['symptoms'])}</p>
              <p style="font-size:0.86rem; color:#9EAF9B; margin:0;"><b>Recommended Intervention:</b> {esc(care['treatment'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Diagnostic Certificate Download
    report = f"""====================================================
LEAF ID — BOTANICAL SCAN READOUT
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
====================================================
Primary Match: {sp} ({cond})
Confidence: {top_p * 100:.2f}%
Status: {'DISEASED' if not healthy else 'HEALTHY'}
Scored Classes: {len(ranked)}
Model Backbone: EfficientNet-B0 (PyTorch CPU)

TOP READOUTS:
""" + "\n".join([f"{i}. {parse_label(l)[0]} ({parse_label(l)[1]}) — {p*100:.2f}%" for i, (l, p) in enumerate(ranked[:5], 1)])

    st.download_button(
        "📄 Download Diagnostic Certificate (.txt)",
        data=report,
        file_name=f"leafid_{sp.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        key="download_report_btn",
    )

with right:
    st.markdown('<div id="leafid-scan" class="anchor-node" aria-hidden="true"></div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-label">SPECIMEN INTAKE</div>', unsafe_allow_html=True)

    intake_tabs = st.tabs(["📁 Upload File", "📷 Live Camera", "🧪 Demo Samples"])
    active_bytes = None
    active_name = None

    with intake_tabs[0]:
        uploaded = st.file_uploader(
            "Drop a leaf photo",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="leaf_scan_file",
        )
        if uploaded is not None:
            active_bytes = uploaded.getvalue()
            active_name = uploaded.name

    with intake_tabs[1]:
        cam_pic = st.camera_input("Capture leaf via camera", label_visibility="collapsed", key="leaf_scan_cam")
        if cam_pic is not None:
            active_bytes = cam_pic.getvalue()
            active_name = "camera_capture.jpg"

    with intake_tabs[2]:
        demo_cols = st.columns(4)
        sample_files = [
            ("sample_mango.jpg", "Mango"),
            ("sample_diseased_lemon.jpg", "Lemon (Canker)"),
            ("sample_aloevera.jpg", "Aloe Vera"),
            ("sample_diseased_tomato.jpg", "Tomato (Blight)"),
        ]
        for idx, (sf, sname) in enumerate(sample_files):
            p = SAMPLES_DIR / sf
            with demo_cols[idx]:
                if p.exists():
                    if st.button(sname, key=f"btn_demo_{idx}", use_container_width=True):
                        st.session_state["demo_sample_path"] = str(p)
                        st.rerun()

        if st.session_state.get("demo_sample_path"):
            dsp = Path(st.session_state["demo_sample_path"])
            if dsp.exists():
                active_bytes = dsp.read_bytes()
                active_name = dsp.name

    handle_specimen(active_bytes, active_name)

# ─────────────────────────────────────────────────────────────────────────────
# 7 · PAGE-LEVEL JS
# ─────────────────────────────────────────────────────────────────────────────
PAGE_JS = """
<script>
(function(){
  function ready(fn){ if(document.readyState!=='loading'){fn();} else {document.addEventListener('DOMContentLoaded',fn);} }
  ready(function(){
    var doc; try{ doc = window.parent.document; }catch(e){ return; }
    if(!doc) return;
    try{
      var hero = doc.getElementById('leafid-hero');
      var bar  = doc.getElementById('leafid-ctabar');
      if(hero && bar && 'IntersectionObserver' in window){
        new IntersectionObserver(function(es){
          for(var i=0;i<es.length;i++){ bar.classList.toggle('is-visible', !es[i].isIntersecting); }
        }, {rootMargin:'-64px 0px 0px 0px', threshold:0}).observe(hero);
      }
    }catch(e){}
    try{
      var v = doc.getElementById('leafid-hero-video'), b = doc.getElementById('leafid-mute');
      if(v && b){ b.addEventListener('click', function(){
        v.muted = !v.muted;
        b.classList.toggle('is-on', !v.muted);
        b.setAttribute('aria-pressed', String(!v.muted));
        b.textContent = v.muted ? 'SOUND · OFF' : 'SOUND · ON';
      }); }
    }catch(e){}
    try{
      var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      doc.addEventListener('click', function(ev){
        var a = ev.target && ev.target.closest ? ev.target.closest('a[data-scroll]') : null;
        if(!a) return;
        var id = a.getAttribute('data-scroll'), t = id && doc.getElementById(id);
        if(t){ ev.preventDefault(); t.scrollIntoView({behavior: reduce ? 'auto' : 'smooth', block:'start'}); }
      });
    }catch(e){}
  });
})();
</script>
"""

COUNTUP_JS = """
<script>
(function(){
  function ready(fn){ if(document.readyState!=='loading'){fn();} else {document.addEventListener('DOMContentLoaded',fn);} }
  ready(function(){
    try{
      var doc = window.parent.document;
      var el = doc.getElementById('leafid-conf-num');
      if(!el) return;
      var target = parseFloat(el.getAttribute('data-target')||'0');
      if(isNaN(target)) return;
      var reduce = false;
      try{ reduce = doc.defaultView.matchMedia('(prefers-reduced-motion: reduce)').matches; }catch(e){}
      if(reduce){ el.textContent = target.toFixed(1) + '%'; return; }
      var t0 = null, dur = 850;
      function step(ts){
        if(t0 === null) t0 = ts;
        var p = Math.min((ts - t0)/dur, 1);
        el.textContent = (target * (1 - Math.pow(1 - p, 3))).toFixed(1) + '%';
        if(p < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }catch(e){}
  });
})();
</script>
"""

st.markdown(
    """
<div class="cta-bar" id="leafid-ctabar" role="navigation" aria-label="Quick actions">
  <div class="cta-inner">
    <span class="cta-brand">LEAF&nbsp;ID</span>
    <span class="cta-hint">%(nc)d plant classes · disease readout on %(nd)d · MIT licensed</span>
    <span class="cta-actions">
      <a class="btn btn-primary" href="#leafid-scan" data-scroll="leafid-scan">Identify a Leaf</a>
      <a class="btn btn-ghost" href="%(repo)s" target="_blank" rel="noopener noreferrer">Star on GitHub</a>
    </span>
  </div>
</div>"""
    % {"nc": N_CLASSES, "nd": N_DISEASE, "repo": REPO_URL},
    unsafe_allow_html=True,
)
components.html(PAGE_JS, height=0)

# ─────────────────────────────────────────────────────────────────────────────
# 8 · SECTION 02 — PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    section_header("02", "SEC.02 · PROTOCOL", "How it works", "Four steps, one rule: the leaf is the specimen."),
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="steps-grid">'
    + "".join(
        '<div class="step reveal" style="--d:%.2fs"><span class="num">%s</span>'
        "<h3>%s</h3><p>%s</p></div>" % (0.10 * i, esc(n), esc(t), esc(d))
        for i, (n, t, d) in enumerate(STEPS)
    )
    + "</div>"
    + '<div class="tip-strip">'
    + "".join('<span class="tip"><b>▸</b> %s</span>' % esc(t) for t in TIPS)
    + "</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 9 · SECTION 03 — CLASS REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    section_header(
        "03",
        "SEC.03 · CLASS REGISTRY",
        "What it recognizes",
        "The full registry, read live from class_names.json — nothing hardcoded.",
    ),
    unsafe_allow_html=True,
)

if not CLASS_NAMES:
    st.markdown(
        '<div class="grid-empty">Class registry not found — place class_names.json next to app.py '
        "to populate this section.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="scan-label">FILTER REGISTRY</div>', unsafe_allow_html=True)
    q = st.text_input(
        "Filter classes",
        value="",
        key="class_search",
        placeholder="e.g. tomato, apple, blight, mint …",
        label_visibility="collapsed",
    )

    REGISTRY = {"fruit": [], "vegetable": [], "medicinal": [], "ornamental": [], "other": []}
    for _raw in CLASS_NAMES:
        REGISTRY[category_for(parse_label(_raw)[0])].append(_raw)

    order = [("all", "All Classes", CLASS_NAMES)] + [
        (k, CATEGORY_LABELS[k], REGISTRY[k])
        for k in ("fruit", "vegetable", "medicinal", "ornamental", "other")
        if REGISTRY[k]
    ]
    tabs = st.tabs(["%s · %d" % (lbl, len(items)) for _, lbl, items in order])
    q_low = q.strip().lower()
    for tab, (key, _lbl, items) in zip(tabs, order):
        with tab:
            if q_low:
                items = [
                    n
                    for n in items
                    if q_low in n.lower()
                    or q_low in parse_label(n)[0].lower()
                    or q_low in parse_label(n)[1].lower()
                ]
            if not items:
                st.markdown(
                    '<div class="grid-empty">No classes match this filter — clear the search or '
                    "pick another category.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(class_grid(items), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 10 · SECTION 04 — TRANSFER LEARNING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    section_header("04", "SEC.04 · TRANSFER LEARNING", "Under the hood", "ImageNet → leaves, in four moves."),
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="pipeline">'
    + "".join(
        '<div class="stage" style="--d:%.2fs"><span class="node">%s</span>'
        "<h3>%s</h3><p>%s</p></div>" % (0.15 * i, esc(n), esc(t), esc(d))
        for i, (n, t, d) in enumerate(PIPELINE)
    )
    + "</div>"
    + '<div class="tech-strip">'
    + "".join('<span class="tech">%s</span>' % esc(t) for t in TECH)
    + "</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 11 · SECTION 05 — SOURCING & TRANSPARENCY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    section_header(
        "05", "SEC.05 · SOURCING", "Data & transparency", "Public data, credited and linked. Honest limits, stated up front."
    ),
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="ds-grid">'
    + "".join(
        '<div class="ds reveal" style="--d:%.2fs"><span class="ds-idx">DS.%02d</span>'
        '<h4>%s</h4><p>%s</p><a class="link" href="%s" target="_blank" rel="noopener noreferrer">'
        "kaggle.com ↗</a></div>" % (0.08 * i, i + 1, esc(n), esc(role), esc(url))
        for i, (n, url, role) in enumerate(KAGGLE_DATASETS)
    )
    + "</div>",
    unsafe_allow_html=True,
)

with st.expander("Licensing notes"):
    st.markdown(
        "- **Code:** MIT License — fork it, audit it, ship it.\n"
        "- **Datasets:** remain under their respective Kaggle licenses; review each dataset page "
        "before redistributing imagery or derived weights.\n"
        "- **This repository** does not re-distribute raw dataset archives; only fine-tuned weights "
        "(`LeafID.pt`) and class indices ship with the application.",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="limits reveal" style="--d:.1s"><span class="num">LIMITATIONS — READ BEFORE SCANNING</span>'
    + "".join("<li>%s</li>" % esc(l) for l in LIMITATIONS)
    + "</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 12 · SECTION 06 — BUILDERS & COMMUNITY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    section_header(
        "06", "SEC.06 · THE PEOPLE AT THE BENCH", "Builders & community", "Two builders, one MIT license, and an open invitation."
    ),
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="builders">'
    + "".join(
        '<div class="builder reveal" style="--d:%.2fs"><div class="avatar" aria-hidden="true">%s</div>'
        '<div><h4>%s</h4><a class="handle" href="%s" target="_blank" rel="noopener noreferrer">%s ↗</a>'
        "<p>Co-author · Leaf-ID</p></div></div>" % (0.08 * i, esc(init), esc(name), esc(url), esc(handle))
        for i, (name, handle, url, init) in enumerate(AUTHORS)
    )
    + "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="contrib">
  <div class="cstep reveal" style="--d:.05s"><span class="num">01 · FORK</span>
    <p>Fork the repository and create a feature branch: <code>git checkout -b feature/your-idea</code></p>
    <a class="link" href="%(fork)s" target="_blank" rel="noopener noreferrer">Open fork page ↗</a></div>
  <div class="cstep reveal" style="--d:.13s"><span class="num">02 · COMMIT</span>
    <p>Keep changes scoped and documented. Small pull requests merge faster.</p>
    <a class="link" href="%(repo)s" target="_blank" rel="noopener noreferrer">Read the source ↗</a></div>
  <div class="cstep reveal" style="--d:.21s"><span class="num">03 · OPEN A PR</span>
    <p>Include a sample scan (before/after) in the description so reviewers can reproduce it.</p>
    <a class="link" href="%(prs)s" target="_blank" rel="noopener noreferrer">Pull requests ↗</a></div>
</div>
<div class="issue-line reveal" style="--d:.26s">Found a misidentification? <a href="%(issues)s" target="_blank" rel="noopener noreferrer">Open an issue ↗</a>
with the photo, the predicted class and the confidence — bad reads are training signals.</div>
"""
    % {"fork": FORK_URL, "repo": REPO_URL, "prs": PRS_URL, "issues": ISSUES_URL},
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 13 · SECTION 07 — RETURN TO BENCH & FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="cta-band reveal" id="leafid-cta-band">
  <div class="cta-eyebrow"><span class="tick"></span>SEC.07 · RETURN TO THE BENCH</div>
  <h2>The lens is warmed up.<br>Bring me a <span class="lme">leaf</span>.</h2>
  <div class="cta-actions">
    <a class="btn btn-primary" href="#leafid-scan" data-scroll="leafid-scan">Identify a Leaf</a>
    <a class="btn btn-ghost" href="%s" target="_blank" rel="noopener noreferrer">Star on GitHub</a>
  </div>
  <p class="cta-note">Scrolls to the scanner at the top of the page.</p>
</div>
<div class="footer">
  <span>LEAF ID · OPEN-SOURCE BOTANICAL IDENTIFICATION</span>
  <span>CODE: MIT · DATA: SEE DATASET LICENSES · BUILT BY RANA UMAR BILAL &amp; MIAN ZAID</span>
</div>"""
    % REPO_URL,
    unsafe_allow_html=True,
)