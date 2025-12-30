# 🔥 AI Wildfire Alert System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent real-time wildfire detection system powered by **YOLOv11 Segmentation** with automatic email alerts. This system uses computer vision and deep learning to detect fires in video streams or uploaded footage, providing instant notifications to help prevent disasters.

![Fire Detection Demo](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🌟 Features

- **🎯 Real-Time Fire Detection**: Uses YOLOv11 segmentation model for accurate fire detection
- **📹 Multiple Input Sources**: Support for live webcam feed and video file uploads (MP4, AVI, MKV)
- **📧 Automatic Email Alerts**: Sends instant email notifications with fire detection snapshots
- **📍 Location Tracking**: Automatically fetches and displays geographic location
- **📊 Detection Dashboard**: Interactive Streamlit dashboard with statistics and detection logs
- **⏱️ Timestamp Overlay**: Real-time timestamp on live detection feed
- **🔄 Object Tracking**: Persistent tracking of detected fire instances
- **📈 Detection Statistics**: Tracks total fire detections and maintains detailed logs

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **YOLOv11** | Fire segmentation and detection |
| **OpenCV** | Video processing and frame manipulation |
| **Streamlit** | Interactive web dashboard |
| **CVZone** | Enhanced visualization |
| **SMTP** | Email alert notifications |
| **NumPy** | Numerical computations |

---

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam (for live detection)
- Gmail account (for email alerts)
- Internet connection (for location services)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Avinash0377/AI-wildfire-alert-system.git
cd AI-wildfire-alert-system
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Email Settings

Open `main.py` and update the email configuration:

```python
sender_email = 'your_email@gmail.com'  # Your Gmail address
sender_password = 'your_app_password'   # Your Google app password
receiver_email = 'recipient@gmail.com'  # Alert recipient email
```

> **Note**: You need to generate a [Google App Password](https://support.google.com/accounts/answer/185833) for SMTP authentication.

---

## 💻 Usage

### Run the Application

```bash
python -m streamlit run main.py
```

### Dashboard Options

1. **Live Detection**: Click "Live Detection" to start real-time fire detection using your webcam
2. **Upload Video**: Use the file uploader to analyze pre-recorded video files
3. **Stop Detection**: Click "Stop Detection" to halt the detection process

---

## 📁 Project Structure

```
AI-wildfire-alert-system/
│
├── main.py              # Main application with Streamlit dashboard
├── best.pt              # Trained YOLOv11 fire detection model
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── videos/              # Sample video files for testing
    ├── vid.mp4
    ├── videoplayback.mp4
    └── videoplayback (1).mp4
```

---

## ⚙️ Configuration

### Email Alert Settings

| Parameter | Description |
|-----------|-------------|
| `sender_email` | Gmail address for sending alerts |
| `sender_password` | Google App Password |
| `receiver_email` | Email address to receive alerts |
| `max_retries` | Maximum retry attempts for failed emails (default: 3) |
| `delay` | Delay between retry attempts in seconds (default: 5) |

### Detection Settings

| Parameter | Description |
|-----------|-------------|
| Frame Skip | Processes every 3rd frame for performance |
| Resolution | Frames resized to 640x360 for faster processing |
| Location Update | Updates location every 60 frames during live detection |

---

## 📸 Screenshots

### Detection Dashboard
The main interface provides:
- Real-time video feed with fire detection overlays
- Detection statistics showing total fires detected
- Detailed detection log with timestamps and locations

### Email Alert
When fire is detected, an email is automatically sent containing:
- Subject: "Fire Detected"
- Attachment: Snapshot of the detected fire

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not loading | Ensure `best.pt` is in the same directory as `main.py` |
| Email not sending | Verify Gmail app password and enable "Less secure apps" |
| Webcam not working | Check camera permissions and try `cv2.CAP_DSHOW` backend |
| Slow detection | Reduce video resolution or increase frame skip rate |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Sudimalla Avinash**

- 📧 Email: [sudhimallaavinash@gmail.com](mailto:sudhimallaavinash@gmail.com)
- 🐙 GitHub: [@Avinash0377](https://github.com/Avinash0377)

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLOv11 model
- [Streamlit](https://streamlit.io/) for the amazing dashboard framework
- [OpenCV](https://opencv.org/) for computer vision capabilities

---

## ⭐ Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

<p align="center">
  Made with ❤️ for wildfire prevention and safety
</p>
