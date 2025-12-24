<p align="center">
  <img src="assets/civisense_banner.png" width="850"/>
</p>

#  CIVISENSE  
### AI-powered Urban Damage Intelligence & Vision Model Health Monitoring

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
  <img width="740" height="271" alt="image" src="https://github.com/user-attachments/assets/361e2c60-6692-49e2-9c3f-d776a5f4d6ac" />

  <img width="1832" height="821" alt="image" src="https://github.com/user-attachments/assets/a5a38053-9b4b-4c87-b2c0-d6514e9952b1" />

</p>

###  API Responses & Drift Metrics

<p align="center">
<img width="574" height="843" alt="image" src="https://github.com/user-attachments/assets/f1c2ac5e-2fb9-4058-bb8b-8000039b3ff0" />
  
<img width="740" height="641" alt="image" src="https://github.com/user-attachments/assets/f1922371-7ea9-46bc-a82c-2cd87e311482" />

<img width="811" height="690" alt="image" src="https://github.com/user-attachments/assets/8584182b-6719-4433-b729-706839da90aa" />


</p>

###  Annotated Detection Results

<p align="center">
  <img width="508" height="345" alt="image" src="https://github.com/user-attachments/assets/3aac1452-d51d-4f30-9bba-6c75befc6ae5" />


</p>

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























