  <img src="assets/civisense_banner.png" width="850"/>
</p>

#  CIVISENSE  
### AI-powered Urban Damage Intelligence & Vision Model Health Monitoring
 > AI-powered urban infrastructure damage detection with severity analysis and model health monitoring.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue"/>
  <img src="https://img.shields.io/badge/YOLOv11-Ultralytics-orange"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-green"/>
  <img src="https://img.shields.io/badge/MongoDB-Atlas-brightgreen"/>
  <img src="https://img.shields.io/badge/Status-Completed-success"/>
</p>

---

##  Overview

**CIVISENSE** is an end-to-end **computer vision and MLOps system** designed to detect road infrastructure damage, evaluate its severity, and continuously monitor model health using drift detection techniques.

The project simulates a **real-world smart city AI pipeline** by integrating:
- deep learning–based vision models
- backend APIs
- statistical model monitoring
- persistent database logging

---

##  Key Features

###  Road Damage Detection
- YOLOv11-based object detection
- Detects potholes, cracks, and surface degradation

###  Severity & Risk Scoring Engine
- Computes severity using confidence & bounding-box area
- Assigns risk levels: **LOW / MEDIUM / HIGH**

###  Model Drift Monitoring
- Tracks:
  - confidence drift
  - bounding-box area drift
  - detection frequency drift
- Provides health status:
  -  STABLE
  -  WARNING
  -  RETRAIN_SUGGESTED

###  FastAPI Backend
- Real-time inference API
- Model health & drift metrics endpoint

###  MongoDB Atlas Integration
- Stores predictions and model-health logs
- Enables analytics, auditing, and monitoring

###  Optional Visualization
- Generates annotated images with bounding boxes
- Useful for debugging, demos, and human validation
- Keeps core API lightweight and scalable


##  System Architecture
Image Input  
      ↓  
YOLOv11 Damage Detector  
      ↓  
Severity & Risk Engine  
      ↓  
MongoDB Atlas  
     ↓  
Drift Monitoring  
     ↓  
FastAPI Endpoints


##  Tech Stack

| Layer | Tools |
|------|------|
| Computer Vision | YOLOv11 (Ultralytics), PyTorch |
| Backend API | FastAPI |
| Database | MongoDB Atlas |
| Model Monitoring | Statistical Drift Detection |

---

##  Dataset & Model

- Trained using:
  - **RDD2022 Road Damage Dataset**
  - **Roboflow Road Damage Dataset**
- Damage categories include potholes, cracks, and surface defects

>  Model weights (`damage_detector.pt`) are excluded from this repository due to size and licensing constraints.

---

## Sample Outputs

<p align="center">
  <img width="1793" height="672" alt="Screenshot 2026-01-10 180104" src="https://github.com/user-attachments/assets/57620c98-51e6-49a7-83c4-73a2682dd050" />

  <img width="1798" height="646" alt="Screenshot 2026-01-10 180142" src="https://github.com/user-attachments/assets/16cc6d8d-aed8-4b05-ae4c-5c993928bb3a" />
  
  <img width="1660" height="840" alt="Screenshot 2026-01-10 180200" src="https://github.com/user-attachments/assets/bec8a027-1c09-4c75-95b7-8d246906fe68" />

   
</p>

###  Streamlit

<p align="center">
<img width="1723" height="466" alt="Screenshot 2026-01-10 180205" src="https://github.com/user-attachments/assets/99ee2fe6-02c6-4263-879c-8fdd11da7c4b" />
  
<img width="1766" height="673" alt="Screenshot 2026-01-10 180215" src="https://github.com/user-attachments/assets/8e9383cb-467f-47f6-9315-248582c9e42c" />


</p>

###  Annotated Detection Results

<p align="center">
  <img width="1660" height="840" alt="Screenshot 2026-01-10 180200" src="https://github.com/user-attachments/assets/72149b7b-ff8f-4cd5-bfe3-0e729771f311" />


</p>

### Analytics (damage summary)

<img width="1630" height="811" alt="Screenshot 2026-01-10 180223" src="https://github.com/user-attachments/assets/844bdf0c-fc7d-4899-87b8-4fa18642b6ca" />

---

##  How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn backend.app:app --reload
# Once running: Go to Swagger UI
Swagger UI: http://127.0.0.1:8000/docs

```
##  Viewing Annotated Images (Bounding Boxes)

CIVISENSE can optionally generate **annotated images** with bounding boxes around detected road damage.

### How it works

- When an image is uploaded to the `/predict` endpoint:
  - The model performs detection
  - Bounding boxes are drawn on the image
  - The annotated image is saved on the server

### How to view the image

1. Upload an image using the `/predict` endpoint  
   (via Swagger UI or API client)

2. The API response will include an `annotated_image` field:

```json
{
  "annotated_image": "/outputs/3f8a2c91e7b44c1b.jpg"
}
Open the image in your browser:


http://127.0.0.1:8000/outputs/3f8a2c91e7b44c1b.jpg
```

 ### API Endpoints
Method	Endpoint	Description

POST	/predict	Detect road damage from uploaded images

GET	/model-health	Retrieve drift metrics & model status

## Engineering Highlights
- Designed an end-to-end computer vision pipeline with production-style APIs and persistent analytics
- Implemented statistical drift detection to monitor real-world model degradation




























