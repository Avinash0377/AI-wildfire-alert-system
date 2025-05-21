# AI-wildfire-alert-system
🔥 Wildfire Detection System using YOLOv11 and U-Net
An AI-powered real-time forest wildfire detection and early warning system using deep learning models (YOLOv11 for object detection and U-Net for segmentation). This project aims to improve wildfire monitoring efficiency with high-speed inference, precision detection, and automated alerting mechanisms.

🚀 Project Overview
Wildfires pose severe risks to ecosystems and human settlements. This project implements a robust, real-time wildfire detection framework using:

YOLOv11 for fire and smoke detection in video streams.

U-Net for precise segmentation of affected areas.

Real-time alert system via email with annotated image evidence.

Streamlit UI for live video feed monitoring and user interaction.

📦 Features
🔍 Real-time detection of fire and smoke in videos

📬 Automated email alerts with frame snapshots

🗺️ Geolocation tagging using IP-based APIs

🎛️ Interactive dashboard built with Streamlit

📊 Performance Metrics
Metric	Value
Precision	93.1%
Recall	91.8%
mAP	92%
F1-Score	0.92
Frame Rate	~30 FPS
Alert Delay	~1–2 sec

🖼️ Sample Outputs
Fire detection with bounding boxes

Live geolocation + timestamp overlays

Email alert snapshot

🧪 Experimental Setup
Software: Python 3.8+, Ultralytics YOLOv11, OpenCV, Streamlit

Hardware: NVIDIA GTX 1070+, 16GB RAM, HD Webcam

Dataset: FLAME Dataset from RoboFlow (1,322 images)


📈 Future Enhancements
Integration with satellite or UAV imagery

Cloud-based dashboard and alert system

Multi-source video feed processing

Expansion to other disaster types (flood, landslide)

⚡ Lightweight, fast inference suitable for deployment on edge devices

🧠 Models Used
YOLOv11 for object detection

Trained on the FLAME Dataset (Fire and Smoke imagery)
