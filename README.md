#  LEAF ID

**LEAF ID** is a deep-learning leaf recognition tool built on an **EfficientNet-B0** architecture. It was trained across a wide, real-world collection of leaves — from common fruit trees and garden vegetables to traditional medicinal and ornamental plants — so it can tell one species from another at a glance, and catch early signs of disease for select species.

 **Live demo:** [huggingface.co/spaces/ranaumarbilal31/leafid](https://huggingface.co/spaces/ranaumarbilal31/leafid)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Overview

Point your camera at a leaf — LEAF ID tells you the species, and for select species, whether it's showing signs of disease.

- **Total Classes:** 84
- **Disease Labels:** 11
- **Architecture:** EfficientNet-B0 (transfer learning)
- **Framework:** PyTorch + torchvision
- **Interface:** Streamlit

---

##  Model

LEAF ID fine-tunes an **EfficientNet-B0** backbone (pretrained on ImageNet) on a combined dataset of plant leaf images, classifying into 84 species categories, 11 of which also carry disease-status labels.

The trained model weights are provided as [`LeadID.pt`](./LeadID.pt), and the full training process — data loading, augmentation, fine-tuning, and evaluation — is documented in the included Jupyter notebook.

---

##  Dataset

This project combines three public Kaggle datasets:

| Dataset | Link |
|---|---|
| 48 Plant Leaves Datasets | [kaggle.com/datasets/developerzulkarnain/48-plant-leaves-datasets](https://www.kaggle.com/datasets/developerzulkarnain/48-plant-leaves-datasets) |
| Plant Leaf Dataset | [kaggle.com/datasets/mahaninghubballi/plant-leaf-dataset](https://www.kaggle.com/datasets/mahaninghubballi/plant-leaf-dataset) |
| Plant Leaves for Image Classification | [kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification](https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification) |

You can download all three programmatically using [`kagglehub`](https://github.com/Kaggle/kagglehub):

```python
import kagglehub

d1 = kagglehub.dataset_download("developerzulkarnain/48-plant-leaves-datasets")
d2 = kagglehub.dataset_download("mahaninghubballi/plant-leaf-dataset")
d3 = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")
```

All rights to the underlying image data belong to their respective dataset authors on Kaggle — please review each dataset's license before reuse.

---

## 📁 Repository Structure

```
Leaf-ID/
├── app.py                 # Streamlit web app
├── LeadID.pt               # Trained EfficientNet-B0 model weights
├── class_names.json       # Mapping of class indices to species/disease labels
├── requirements.txt       # Python dependencies
├── training_notebook.ipynb # Model training & evaluation notebook
├── LICENSE
└── README.md
```

##  Installation

```bash
git clone https://github.com/ranaumarbilal31/Leaf-ID.git
cd Leaf-ID
pip install -r requirements.txt
streamlit run app.py
```

---

## How to  Use:

1. Launch the app with `streamlit run app.py` (or use the [hosted demo](https://huggingface.co/spaces/ranaumarbilal31/leafid)).
2. Upload a JPG or PNG image of a single leaf.
3. LEAF ID returns the predicted species and, for supported species, a disease-status prediction with confidence scores.

---

##  Training

The full training pipeline — dataset merging, preprocessing, augmentation, EfficientNet-B0 fine-tuning, and evaluation — is available in the training notebook included in this repo. Open it in Jupyter or Google Colab to reproduce or extend the model.

---

##  Tech Stack

- **PyTorch** / **torchvision** — model architecture & inference
- **EfficientNet-B0** — backbone architecture
- **Streamlit** — web interface
- **Pillow / NumPy / Pandas** — image and data handling
- **Kaggle / kagglehub** — dataset sourcing

---

##  Contributing

Contributions are welcome! To contribute:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Open a pull request

---

##  License

This project is licensed under the [MIT License](./LICENSE).

---
##  Contact and Acknowledgements

**Rana Umar Bilal**  
🔗 GitHub: [@ranaumarbilal31](https://github.com/ranaumarbilal31)

 **Mian Zaid**  
🔗 GitHub: [@zaid-mian](https://github.com/https://github.com/zaid-mian) 


- Dataset authors on Kaggle (see [Dataset](#-dataset) section above)
- EfficientNet authors (Tan & Le, 2019)

---

<div align="center">

**If you found this project helpful, please give it a star!** ⭐

<sub>Built with Love and a lot of Hardwork</sub>

</div>

