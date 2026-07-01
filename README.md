# 🍃 Leaf-ID

<div align="center">

![Leaf-ID Banner](https://img.shields.io/badge/Leaf--ID-Plant%20Species%20Recognition-success?style=for-the-badge&logo=leaf&logoColor=white&labelColor=2ea44f&color=4CAF50)

**An AI-powered leaf identification system that classifies plant species from leaf images using deep learning.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-red?style=flat-square&logo=keras&logoColor=white)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Issues](https://img.shields.io/github/issues/ranaumarbilal31/Leaf-ID?style=flat-square)](https://github.com/ranaumarbilal31/Leaf-ID/issues)

</div>

---

##  -About the Project

**Leaf-ID** is a deep learning–based computer vision project designed to **identify plant species** from images of their leaves. By leveraging convolutional neural networks (CNNs), the system analyzes leaf shape, texture, and venation patterns to classify them into their respective plant species.

This project has applications in:
-  **Botany & agriculture** — rapid plant identification in the field
-  **Ecology & conservation** — biodiversity monitoring
-  **Education** — helping students and enthusiasts learn about flora
-  **Pharmaceuticals** — identifying medicinal plants

---

##  Features

| Feature | Description |
|---------|-------------|
| 🔍 **Image Classification** | Classifies leaf images into multiple plant species |
| 🧠 **Deep CNN Model** | Built with TensorFlow / Keras for high accuracy |
| 📷 **Image Preprocessing** | Resizing, normalization, and augmentation pipeline |
| 🌐 **Web Interface** | Upload a leaf image and get instant predictions |
| 📊 **Visualization** | Training curves, confusion matrix, and sample predictions |

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

---

##  Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ranaumarbilal31/Leaf-ID.git
cd Leaf-ID
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Linux/macOS
# venv\Scripts\activate         # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Simply Run
```bash
streamlit run app.py
```


---
## 🧠 Model Architecture

The model is a **Convolutional Neural Network (CNN)** built with Keras' Sequential API:

```
┌─────────────────────────────────────┐
│  Input Layer  (224 × 224 × 3)       │
├─────────────────────────────────────┤
│  Conv2D + ReLU  →  MaxPooling2D     │
│  Conv2D + ReLU  →  MaxPooling2D     │
│  Conv2D + ReLU  →  MaxPooling2D     │
│  Conv2D + ReLU  →  MaxPooling2D     │
├─────────────────────────────────────┤
│  Flatten                            │
│  Dense + ReLU  →  Dropout           │
│  Dense + ReLU  →  Dropout           │
│  Dense + Softmax  (num_classes)     │
└─────────────────────────────────────┘
```

- **Optimizer:** Adam
- **Loss:** Categorical Cross-Entropy
- **Metrics:** Accuracy, Precision, Recall

---

##  Results

| Metric          | Value   |
|-----------------|---------|
| ✅ Training Accuracy     | ~ 98 %  |
| ✅ Validation Accuracy   | ~ 94 %  |
| ✅ Test Accuracy         | ~ 93 %  |
| 📉 Training Loss         | ~ 0.04  |
| 📉 Validation Loss       | ~ 0.18  |

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. 🍴 Fork the project
2. 🌿 Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 📬 Open a Pull Request

Please make sure to update tests as appropriate and follow the existing code style.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

## 📧 Contact

**Rana Umar Bilal**  
🔗 GitHub: [@ranaumarbilal31](https://github.com/ranaumarbilal31)

 **Mian Zaid**  
🔗 GitHub: [@zaid-mian](https://github.com/https://github.com/zaid-mian) 

---

<div align="center">

**If you found this project helpful, please give it a star!** ⭐

<sub>Built with Love and a lot of Hardwork</sub>

</div>
```
