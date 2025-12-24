<p align="center">
  <img src="assets/civisense_banner.png" width="850"/>
</p>

# 🚦 CIVISENSE  
### AI-powered Urban Damage Intelligence & Vision Model Health Monitoring

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue"/>
  <img src="https://img.shields.io/badge/YOLOv11-Ultralytics-orange"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-green"/>
  <img src="https://img.shields.io/badge/MongoDB-Atlas-brightgreen"/>
  <img src="https://img.shields.io/badge/Status-Completed-success"/>
</p>

---

## 📌 Overview

**CIVISENSE** is an end-to-end **computer vision and MLOps system** designed to detect road infrastructure damage, evaluate its severity, and continuously monitor model health using drift detection techniques.

The project simulates a **real-world smart city AI pipeline** by integrating:
- deep learning–based vision models
- backend APIs
- statistical model monitoring
- persistent database logging

---

## 🚀 Key Features

### 🛣️ Road Damage Detection
- YOLOv11-based object detection
- Detects potholes, cracks, and surface degradation

### ⚠️ Severity & Risk Scoring Engine
- Computes severity using confidence & bounding-box area
- Assigns risk levels: **LOW / MEDIUM / HIGH**

### 🧠 Model Drift Monitoring
- Tracks:
  - confidence drift
  - bounding-box area drift
  - detection frequency drift
- Provides health status:
  - 🟢 STABLE
  - 🟡 WARNING
  - 🔴 RETRAIN_SUGGESTED

### 🔌 FastAPI Backend
- Real-time inference API
- Model health & drift metrics endpoint

### 🗄️ MongoDB Atlas Integration
- Stores predictions and model-health logs
- Enables analytics, auditing, and monitoring

### 🖼️ Optional Visualization
- Generates annotated images with bounding boxes
- Useful for debugging, demos, and human validation
- Keeps core API lightweight and scalable

---

## 🧱 System Architecture

Image Input
↓
YOLOv11 Damage Detector
↓
Severity & Risk Engine
↓
MongoDB Atlas
↓
Drift Monitoring Module
↓
FastAPI Endpoints

yaml
Copy code

---

## 🛠️ Tech Stack

| Layer | Tools |
|------|------|
| Computer Vision | YOLOv11 (Ultralytics), PyTorch |
| Backend API | FastAPI |
| Database | MongoDB Atlas |
| Model Monitoring | Statistical Drift Detection |
| Visualization (Optional) | OpenCV, React, Plotly |

---

## 📊 Dataset & Model

- Trained using:
  - **RDD2022 Road Damage Dataset**
  - **Roboflow Road Damage Dataset**
- Damage categories include potholes, cracks, and surface defects

> ⚠️ Model weights (`damage_detector.pt`) are excluded from this repository due to size and licensing constraints.

---

## 🧪 Sample Outputs

### 🔍 API Responses & Drift Metrics
<p align="center">
  <img width="900" src="https://github.com/user-attachments/assets/d2cfd762-5b09-49cf-8489-dc3ebf7e4ea9"/>
  <img width="900" src="https://github.com/user-attachments/assets/76e83b68-07ef-4fb0-8ada-e4a3bb6848f1"/>
</p>

### 🖼️ Annotated Detection Results
<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/8bcb318b-52e4-4043-b31a-c9f11f7c1158"/>
  <img width="700" src="https://github.com/user-attachments/assets/687b8c00-3259-4d5b-b9c1-62672f5e1f8a"/>
</p>

---

## ▶️ How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn backend.app:app --reload
Once running:

Swagger UI: http://127.0.0.1:8000/docs

🔗 API Endpoints
Method	Endpoint	Description
POST	/predict	Detect road damage from uploaded images
GET	/model-health	Retrieve drift metrics & model status
