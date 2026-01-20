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
main.py
│
├── load_model()
├── preprocess_image()
├── classify_image()
└── main()
    ├── Upload & Classify Page
    ├── Gallery Page
    └── Search & Export Page
```

### Key Design Decisions

* **Cached Model Loading**: Ensures the model loads once per session
* **Session State Management**: Persists gallery data without a database
* **Dynamic File Uploader Reset**: Prevents duplicate uploads
* **Separation of Logic & UI**: Easier maintenance and extensibility

---

## 📄 Code Walkthrough

### 1️⃣ Model Loading

```python
@st.cache_resource
def load_cached_model():
    return MobileNetV2(weights="imagenet")
```
* Uses Streamlit caching to avoid reloading the model
* Improves performance and responsiveness

---

### 2️⃣ Image Preprocessing

```python
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img
```

* Converts image to NumPy array
* Resizes to model input size
* Applies ImageNet normalization
* Adds batch dimension

---

### 3️⃣ Image Classification

```python
def classify_image(model, image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return decode_predictions(predictions, top=5)[0]
```

* Runs inference
* Decodes top 5 predictions
* Returns label + confidence scores

---

### 4️⃣ Streamlit Pages

#### 🔹 Upload & Classify

* Upload multiple images
* Progress bar for batch processing
* Immediate AI tagging

#### 🔹 Gallery

* Grid-based image layout
* Expandable tag details
* Image removal support

#### 🔹 Search & Export

* Keyword search across AI tags
* CSV export of predictions
* Data preview using Pandas

---

## 📊 Exported Data Format

The exported CSV includes:
| Column    | Description             |
| --------- | ----------------------- |
| ID        | Unique image identifier |
| Name      | Image filename          |
| Timestamp | Upload time             |
| Label 1–5 | AI predicted classes    |
| Score 1–5 | Confidence percentages  |

This format is suitable for:
* Dataset analysis
* ML auditing
* Reporting & visualization

---


