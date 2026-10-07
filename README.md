# 🌿 LEAF ID Pro

**LEAF ID Pro** is an open-source botanical artificial intelligence platform powered by an **EfficientNet-B0** deep learning model. It was trained across real-world collections of plant foliage — spanning common agricultural crops, fruit trees, garden vegetables, and traditional medicinal plants — enabling it to identify plant taxonomy at a glance and catch early signs of foliar disease with clinical precision.

🌐 **Live Demo:** [huggingface.co/spaces/ranaumarbilal31/leafid](https://huggingface.co/spaces/ranaumarbilal31/leafid)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Key Features

- **84 Botanical Categories:** Recognizes a vast catalog of agricultural staples, tropical fruit trees, medicinal plants, and garden flora.
- **Foliar Pathology Detection:** Detects bacterial cankers, fungal blights, spots, and rusts for key commercial crops.
- **Dual Ingestion Engine:** Upload image files (`JPG`, `JPEG`, `PNG`, `WEBP`) or capture directly via **Live Camera** on desktop or mobile.
- **One-Click Demo Samples:** Instantly test pre-loaded samples (Healthy Mango, Diseased Lemon, Aloe Vera, Diseased Tomato) with a single click.
- **Actionable Botanical Advisory:** Diagnostic breakdown with condition profiles, symptom descriptions, and organic/cultural intervention steps.
- **Diagnostic Certificate Generator:** Export and download timestamped diagnostic health certificates (.txt) for orchard records.
- **100% Offline Capable:** Runs entirely on CPU with zero cloud latency or privacy exposure.

---

## 📁 Repository Structure

```
Leaf-ID/
├── .streamlit/
│   └── config.toml           # Streamlit server, theme, and cross-origin iframe configuration
├── samples/                  # Pre-loaded sample leaf images for instant testing
│   ├── sample_mango.jpg
│   ├── sample_diseased_lemon.jpg
│   ├── sample_aloevera.jpg
│   └── sample_diseased_tomato.jpg
├── app.py                    # Streamlit web application & inference engine
├── LeafID.pt                 # Trained EfficientNet-B0 model weights (~49.8 MB)
├── class_names.json          # Mapping of class indices to species and pathology labels
├── Dockerfile                # Production Docker container configuration
├── requirements.txt          # Python dependencies
├── training_notebook.ipynb   # Model training & fine-tuning notebook
├── LICENSE                   # MIT License
└── README.md
```

---

## 💻 Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/ranaumarbilal31/Leaf-ID.git
cd Leaf-ID

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app.py
```

### Running with Docker

```bash
docker build -t leafid-pro .
docker run -p 8501:8501 leafid-pro
```

---

## 🧠 Model Architecture

LEAF ID fine-tunes an **EfficientNet-B0** convolutional neural network backbone using transfer learning:
- **Input Size:** 224 × 224 pixels (RGB)
- **Normalization:** ImageNet mean `[0.485, 0.456, 0.406]` & std `[0.229, 0.224, 0.225]`
- **Head:** Linear classifier projecting 1,280 features to 84 class logits
- **Activation:** Softmax probability distribution

---

## 📊 Dataset Sources

This project combines three public Kaggle datasets:

| Dataset | Link |
|---|---|
| 48 Plant Leaves Datasets | [kaggle.com/datasets/developerzulkarnain/48-plant-leaves-datasets](https://www.kaggle.com/datasets/developerzulkarnain/48-plant-leaves-datasets) |
| Plant Leaf Dataset | [kaggle.com/datasets/mahaninghubballi/plant-leaf-dataset](https://www.kaggle.com/datasets/mahaninghubballi/plant-leaf-dataset) |
| Plant Leaves for Image Classification | [kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification](https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification) |

---

## 🛠️ Tech Stack

- **PyTorch** & **torchvision** — Deep learning model architecture and inference
- **Streamlit** — Interactive responsive web UI
- **Pillow** & **NumPy** — High-performance image processing
- **Pandas** — Tabular statistics and diagnostic reporting

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

---

## 👥 Authors & Acknowledgements

**Rana Umar Bilal**  
🔗 GitHub: [@ranaumarbilal31](https://github.com/ranaumarbilal31)  
🤗 Hugging Face: [@ranaumarbilal31](https://huggingface.co/ranaumarbilal31)

**Muhammad Zaid Tahir**  
🔗 GitHub: [@zaid-mian](https://github.com/zaid-mian)

- Dataset authors on Kaggle
- EfficientNet authors (Tan & Le, 2019)
