# ─────────────────────────────────────────────────────────────────────────────
# LEAF ID · app.py — field-lab UI
# Stack: Streamlit + PyTorch/torchvision (EfficientNet-B0) + Pillow.
# Run:   streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

import base64
import hashlib
import html as _html
import io
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

# ─────────────────────────────────────────────────────────────────────────────
# 0 · PAGE CONFIG & PATH RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

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

CONFIDENCE_FLOOR = 0.45
MAX_PREVIEW_PX = 640
REPO_URL = "https://github.com/ranaumarbilal31/Leaf-ID"
ISSUES_URL = REPO_URL + "/issues"
PRS_URL = REPO_URL + "/pulls"
FORK_URL = REPO_URL + "/fork"

_FAVICON = ASSETS / "favicon.svg"
if not _FAVICON.exists() and (ROOT.parent / "assets" / "favicon.svg").exists():
    _FAVICON = ROOT.parent / "assets" / "favicon.svg"

st.set_page_config(
    page_title="LEAF ID — AI leaf identification",
    page_icon=str(_FAVICON) if _FAVICON.exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1 · CLASS REGISTRY & PARSER
# ─────────────────────────────────────────────────────────────────────────────
def load_class_names(path):
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


SPECIES_STATES = {}
for _n in CLASS_NAMES:
    _sp, _cd, _h = parse_label(_n)
    SPECIES_STATES.setdefault(_sp, set()).add(_cd)

DISEASE_SPECIES = {sp for sp, cds in SPECIES_STATES.items() if "Diseased" in cds}
N_DISEASE = len(DISEASE_SPECIES) if DISEASE_SPECIES else 11

BOTANICAL_CARE = {
    "Lemon": {
        "condition": "Citrus Canker / Bacterial Blight",
        "symptoms": "Raised corky lesions surrounded by yellow chlorotic halos on foliage and twigs.",
        "treatment": "Prune severely affected twigs; apply copper-based bactericide spray; avoid overhead watering.",
    },
    "Mango": {
        "condition": "Anthracnose / Leaf Blight",
        "symptoms": "Dark brown, angular to irregular necrotic spots that coalesce into blights.",
        "treatment": "Prune canopy to improve air circulation; apply neem oil or copper oxychloride fungicide.",
    },
    "Tomato": {
        "condition": "Early Blight / Septoria Leaf Spot",
        "symptoms": "Concentric target-like brown spots with yellow halos, starting on lower leaves.",
        "treatment": "Remove lower foliage within 12 inches of soil; mulch base heavily; apply organic copper fungicide.",
    },
    "Guava": {
        "condition": "Guava Wilt / Anthracnose",
        "symptoms": "Curling, yellowing leaves with rusty-brown spots leading to premature defoliation.",
        "treatment": "Drench root zone with Trichoderma viride; ensure soil drainage; apply balanced micronutrients.",
    },
    "Pomegranate": {
        "condition": "Bacterial Blight / Cercospora Spot",
        "symptoms": "Water-soaked dark brown spots turning black with chlorotic margins.",
        "treatment": "Prune during dry spells with sterilized shears; apply copper hydroxide spray.",
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

# ─────────────────────────────────────────────────────────────────────────────
# 2 · MODEL / INFERENCE CORE
# ─────────────────────────────────────────────────────────────────────────────
def _torch_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        pass
    except Exception:
        pass
    return torch.load(str(path), map_location="cpu")


@st.cache_resource(show_spinner=False)
def load_model():
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

def preview_data_uri(image_bytes):
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

FALLBACK_CSS = """
html{scroll-behavior:smooth}
body{background:#07100B;color:#E9F2E4}
.stApp{background:transparent;color:#E9F2E4}
header[data-testid="stHeader"],footer[data-testid="stFooter"],#MainMenu,
[data-testid="stSidebar"],[data-testid="stToolbar"],#stDecoration{display:none!important}
"""

def inject_css():
    css = FALLBACK_CSS
    try:
        css += STYLE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass
    st.markdown("<style>%s</style>" % css, unsafe_allow_html=True)

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

# ─────────────────────────────────────────────────────────────────────────────
# 4 · THEME & BACKGROUND STACK
# ─────────────────────────────────────────────────────────────────────────────
inject_css()
st.session_state.setdefault("scan_seq", 0)
st.session_state.setdefault("scan_digest", None)
st.session_state.setdefault("demo_sample_path", None)

stack_html = '<div class="bg-stack" aria-hidden="true"><div class="bg-veins"></div><div class="bg-shade"></div><div class="bg-grain"></div></div>'
st.markdown(stack_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5 · MAIN SPLIT LAYOUT: HEADLINE ON LEFT | UPLOAD BLOCK ON RIGHT
# ─────────────────────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.markdown(
        '<div class="inline-alert err">MODEL OFFLINE — LeafID.pt was not found. '
        'Please ensure LeafID.pt exists in the application root directory.</div>',
        unsafe_allow_html=True,
    )

# Two-column layout: Left is Headline & Specs; Right is Specimen Intake & Output
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown(
        """
<div class="hero-clean">
  <div class="eyebrow"><span class="tick"></span>OPEN-SOURCE BOTANICAL SCAN BENCH · MIT LICENSED</div>
  <h1 class="display">Put a leaf<br>under the <span class="lme">lens</span>.</h1>
  <p class="hero-sub">LEAF ID reads a single leaf photo and returns species, disease status
  where supported, and ranked confidence — a fine-tuned EfficientNet-B0 you can audit line by line.</p>
  <div class="chip-row">
    <span class="chip"><b>%(nclasses)d</b> PLANT CLASSES</span>
    <span class="chip"><b>%(ndisease)d</b> DISEASE READOUTS</span>
    <span class="chip"><b>B0</b> EFFICIENTNET BACKBONE</span>
    <span class="chip"><b>MIT</b> OPEN LICENSE</span>
  </div>
</div>"""
        % {"nclasses": N_CLASSES, "ndisease": N_DISEASE},
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Classes indexed", "%02d" % N_CLASSES)
    with m2:
        st.metric("Disease readouts", "%02d" % N_DISEASE)
    with m3:
        st.metric("Input plate", "224×224")

with col_right:
    st.markdown('<div id="leafid-scan" class="anchor-node" aria-hidden="true"></div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-label">SPECIMEN INTAKE</div>', unsafe_allow_html=True)

    intake_tabs = st.tabs(["Upload File", "Demo Samples"])
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
        demo_cols = st.columns(4)
        sample_files = [
            ("sample_tomato_healthy.jpg", "Tomato (Healthy)"),
            ("sample_tomato_blight.jpg", "Tomato (Late Blight)"),
            ("sample_potato_healthy.jpg", "Potato (Healthy)"),
            ("sample_grape_healthy.jpg", "Grape (Healthy)"),
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

    # Handle specimen readout
    if not MODEL_PATH.exists():
        st.markdown(empty_state("MODEL OFFLINE — LeafID.pt not found"), unsafe_allow_html=True)
    elif not active_bytes:
        st.markdown(empty_state(), unsafe_allow_html=True)
    else:
        try:
            Image.open(io.BytesIO(active_bytes)).verify()
            is_valid = True
        except Exception:
            is_valid = False

        if not is_valid:
            st.markdown(
                '<div class="inline-alert err">INVALID SPECIMEN — %s is not a readable image.</div>' % esc(active_name),
                unsafe_allow_html=True,
            )
        elif not CLASS_NAMES:
            st.markdown(empty_state("CLASS REGISTRY MISSING — class_names.json not found"), unsafe_allow_html=True)
        else:
            uri, size = preview_data_uri(active_bytes)
            prog = progress_step(10, "SCANNING SPECIMEN …")
            time.sleep(0.08)
            try:
                pset_ok = True
                prog.progress(40, text="MATCHING VEIN SIGNATURES …")
            except TypeError:
                prog.progress(40)
                pset_ok = False
            try:
                ranked = run_inference(active_bytes)
            except Exception as e:
                prog.empty()
                st.markdown(
                    '<div class="inline-alert err">INFERENCE FAILED: %s. Try another photo of a single leaf.</div>' % esc(e),
                    unsafe_allow_html=True,
                )
                ranked = None

            if ranked:
                if pset_ok:
                    prog.progress(100, text="READOUT COMPLETE")
                else:
                    prog.progress(100)
                time.sleep(0.10)
                prog.empty()

                digest = hashlib.md5(active_bytes).hexdigest()
                if st.session_state.get("scan_digest") != digest:
                    st.session_state["scan_digest"] = digest
                    st.session_state["scan_seq"] = st.session_state.get("scan_seq", 0) + 1
                    new_scan = True
                else:
                    new_scan = False

                st.markdown(
                    specimen_card(active_name, uri, size, ranked, st.session_state.get("scan_seq", 1), new_scan),
                    unsafe_allow_html=True,
                )

                # Pathology advisory if diseased
                top_label, top_p = ranked[0]
                sp, cond, healthy = parse_label(top_label)
                if not healthy and sp in BOTANICAL_CARE:
                    care = BOTANICAL_CARE[sp]
                    st.markdown(
                        f"""
                        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 18px 20px; margin-top: 14px;">
                          <div style="font-family:'JetBrains Mono', monospace; font-size:0.76rem; color:#EF4444; letter-spacing:0.08em; font-weight:700;">
                            PATHOLOGY ADVISORY · {esc(care['condition'].upper())}
                          </div>
                          <p style="font-size:0.86rem; color:#E9F2E4; margin:8px 0 6px 0;"><b>Observed Symptoms:</b> {esc(care['symptoms'])}</p>
                          <p style="font-size:0.86rem; color:#9EAF9B; margin:0;"><b>Recommended Intervention:</b> {esc(care['treatment'])}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

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
                    "Download Diagnostic Certificate (.txt)",
                    data=report,
                    file_name=f"leafid_{sp.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    key="download_report_btn",
                )

# ─────────────────────────────────────────────────────────────────────────────
# 6 · AUTHORS & LINKS (Simple names and links, no developer icons)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border: none; height: 1px; background: rgba(255,255,255,0.08); margin: 64px 0 32px 0;'>", unsafe_allow_html=True)

st.markdown(
    """
<div class="authors-simple">
  <div class="sec-eyebrow"><span class="tick"></span>AUTHORS & REPOSITORY</div>
  <div class="authors-links">
    <span><b>Rana Umar Bilal</b> — <a href="https://github.com/ranaumarbilal31" target="_blank" rel="noopener noreferrer">@ranaumarbilal31</a></span>
    <span><b>Muhammad Zaid Tahir</b> — <a href="https://github.com/zaid-mian" target="_blank" rel="noopener noreferrer">@zaid-mian</a></span>
    <span><b>Repository</b> — <a href="https://github.com/ranaumarbilal31/Leaf-ID" target="_blank" rel="noopener noreferrer">GitHub</a></span>
    <span><b>Issues</b> — <a href="https://github.com/ranaumarbilal31/Leaf-ID/issues" target="_blank" rel="noopener noreferrer">Report</a></span>
    <span><b>License</b> — MIT</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 7 · SIMPLE FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="footer-simple">
  <span>LEAF ID · OPEN-SOURCE BOTANICAL IDENTIFICATION</span>
  <span>BUILT BY RANA UMAR BILAL &amp; MIAN ZAID · MIT LICENSED</span>
</div>
""",
    unsafe_allow_html=True,
)