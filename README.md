 **CIVISENSE**
AI-Powered Urban Damage Intelligence & Vision Model Health Monitoring

CIVISENSE is an end-to-end computer vision system designed to detect urban road infrastructure damage, assess its severity, and continuously monitor model performance using drift detection techniques.

The project combines computer vision, backend APIs, model monitoring, and database logging to simulate a real-world smart-city analytics pipeline.

 **Features**

 Road Damage Detection using YOLOv11
Detects potholes, cracks, and surface damage from road images

 Severity & Risk Scoring Engine
Assigns impact-based severity scores and risk levels (LOW / MEDIUM / HIGH)

 **FastAPI Backend**
Real-time inference and health monitoring through REST APIs

 **Model Drift Monitoring**
Detects confidence, area, and frequency drift in predictions

 **MongoDB Atlas Integration**
Stores inference results and model-health metrics for analytics & auditing

 **System Architecture**
``` 
Image Input
    ↓
YOLOv11 Damage Detector
    ↓
Severity & Risk Engine
    ↓
MongoDB Atlas
    ↓
Model Drift Monitor
    ↓
FastAPI Endpoints
```

 **Tech Stack**
Core Technologies
Computer Vision: YOLOv11 (Ultralytics), PyTorch
Backend API: FastAPI
Database: MongoDB Atlas (Cloud)
Model Monitoring: Statistical Drift Analysis

**Dataset & Model**

Trained using RDD2022 and Roboflow Road Damage datasets
Dataset includes potholes, cracks, and surface damage
Model weights (damage_detector.pt) are excluded from the repository due to size and licensing constraints

 How to Run Locally
# Install dependencies
```
pip install -r requirements.txt

# Start FastAPI server
uvicorn backend.app:app --reload
```
Once running:

Open API docs at: http://127.0.0.1:8000/docs

🔗 API Endpoints
Method	Endpoint	Description
POST	/predict	Run damage detection on uploaded images
GET	/model-health	Retrieve model drift & health status

👤 Author
Devansh Prasad
AI / ML | Computer Vision | Backend Systems

