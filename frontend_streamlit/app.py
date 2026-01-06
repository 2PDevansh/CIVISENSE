import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="CIVISENSE Dashboard",
    layout="wide"
)

st.title(" CIVISENSE")
st.caption("AI-powered Urban Damage Intelligence & Model Health Monitoring")

tab1, tab2, tab3 = st.tabs(
    [" Damage Detection", " Model Health", " Analytics"]
)

# -------------------------------
# TAB 1: DAMAGE DETECTION
# -------------------------------
with tab1:
    st.subheader("Upload Road Image")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            with st.spinner("Running YOLO inference..."):
                files = {
                    "image": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    files=files
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success(
                        f"Detections Found: {result['num_detections']}"
                    )

                    if result["detections"]:
                        df = pd.DataFrame(result["detections"])
                        st.dataframe(df)

                        if "annotated_image" in result:
                            st.image(
                                f"{BACKEND_URL}{result['annotated_image']}",
                                caption="Annotated Output",
                                use_column_width=True
                            )
                else:
                    st.error("Inference failed")

# -------------------------------
# TAB 2: MODEL HEALTH
# -------------------------------
with tab2:
    st.subheader("Vision Model Health")

    if st.button("Check Model Health"):
        with st.spinner("Computing drift metrics..."):
            response = requests.get(
                f"{BACKEND_URL}/model-health"
            )

            if response.status_code == 200:
                health = response.json()

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Drift Score",
                    round(health["drift_score"], 3)
                )
                col2.metric(
                    "Confidence Drift",
                    round(health["confidence_drift"], 3)
                )
                col3.metric(
                    "Area Drift",
                    round(health["area_drift"], 3)
                )

                st.metric(
                    "Detection Frequency Drift",
                    round(health["frequency_drift"], 3)
                )

                status = health["status"]
                if status == "STABLE":
                    st.success(" Model Stable")
                elif status == "WARNING":
                    st.warning(" Drift Warning")
                else:
                    st.error(" Retraining Suggested")
            else:
                st.error("Failed to fetch model health")

# -------------------------------
# TAB 3: ANALYTICS
# -------------------------------
with tab3:
    st.subheader("Damage Analytics")

    response = requests.get(
        f"{BACKEND_URL}/analytics/summary"
    )

    if response.status_code == 200:
        data = response.json()

        col1, col2 = st.columns(2)
        col1.metric(
            "Images Processed",
            data["total_images_processed"]
        )
        col2.metric(
            "High-Risk Detections",
            data["high_risk_detections"]
        )

        if data["damage_distribution"]:
            df = pd.DataFrame(data["damage_distribution"])
            df.rename(columns={"_id": "Damage Type"}, inplace=True)

            st.subheader("Damage Distribution")
            st.bar_chart(
                df.set_index("Damage Type")
            )
    else:
        st.error("Failed to load analytics")
