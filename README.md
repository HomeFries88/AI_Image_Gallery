# 📷 AI Image Gallery Manager

An end to end **AI Powered Image Classification and Gallery Management Application** build with **Streamlit** and **Tensorflow (MobileNetV2)**.
This project allows users to upload multiple images, automatically classify them using a pre-trained deep learning model, organize them into a searchable gallery, and export AI generated files into a csv file.

This repository is used to demonstrate **practical machine learning integration**, **clean application architecture**, and **user focused UI design**.

---

## 🚀 Project Overview

The **AI Image Gallery Manager** is a web-based application that:

* Accepts **multiple image uploads**
* Uses a **pre-trained convolutional neural network (MoblileNetV2)** to classify images
* Automatically assigns **AI generated tags**
* Stores images and predictions into an interactive gallery
* Allows **searching images by AI tags**
* Exports classification results to **csv** for futher analysis

The project emphasizes:

* Real world ML inference
* Efficient state management in Streamlit
* Scalable, modular Python code
* Clear separation of concerns (model loading, pre-processing, and user interface)

---

## 🧠 AI and Machine Learning Details

### Model Used:

* **MobileNetV2** (from `tensorflow.keras.applications`)
* Pre-trained on **ImageNet**

### Why MobileNetV2?

* Lightweight and fast
* Optimized for real-time inference
* Ideal for interactive applications

### Inference Flow

1. Images resized to **224x224**
2. Pixel values are normalized using `preprocess_input`
3. Images passed through **C.N.N.**
4. Top **5 Predictions** are decoded and stored

Each image stores:

* Predicted labels
* Confident scores
* Timestamp of upload

---

## ⚒️ Technologies Used:

### Core Stack:

* **Python 3** - main programming language
* **Streamlit** - interactive web interface
* **Tensorflow/Keras** - deep learning inference
* **OpenCV (CV2)** - image pre-processing
* **Numpy** - numerical operations
* **Pandas** - data manipulation and CSV exports
* **PIL** - image handling

### Why Streamlit?

* Rapid prototyping
* Minimal boilerplate
* Built-in session state
* Perfect for ML demos and dashboards

---

## 🧩 Application Architecture

```
app.py
│
├── load_model()
├── preprocess_image()
├── classify_image()
└── main()
    ├── Upload & Classify Page
    ├── Gallery Page
    └── Search & Export Page
```