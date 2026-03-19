# AI Wildfire Alert System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11--Seg-orange.svg)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent real-time wildfire detection system powered by **YOLOv11 Segmentation** with automatic email alerts. Uses computer vision and deep learning to detect and segment fire regions in video streams or uploaded footage, with instant email notifications to help prevent disasters.

---

## Features

- **Real-Time Fire Detection** — YOLOv11 segmentation model with CLAHE-enhanced preprocessing
- **Segmentation Masks** — Visual fire region overlay with color-coded bounding boxes
- **Multiple Input Sources** — Webcam, RTSP/IP camera streams, and video file uploads
- **Automatic Email Alerts** — HTML email with fire snapshot, location, timestamp, and confidence
- **Temporal Confirmation** — Requires 3 consecutive fire frames to eliminate false positives
- **Location Tracking** — Auto-fetches geographic location via IP-based API
- **Configurable Settings** — Sidebar controls for confidence threshold, email cooldown, and more
- **Detection Dashboard** — Interactive Streamlit dashboard with statistics and detection logs
- **Modular Architecture** — Clean separation of concerns across multiple files

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **YOLOv11-Seg** | Fire segmentation and detection |
| **OpenCV** | Video processing, CLAHE preprocessing |
| **Streamlit** | Interactive web dashboard |
| **CVZone** | Enhanced bounding box visualization |
| **SMTP** | HTML email alert notifications |
| **NumPy** | Numerical computations |

---

## Project Structure

```
AI-wildfire-alert-system/
├── main.py                  # Streamlit app — UI and detection loop
├── config.py                # All constants and default values
├── utils/
│   ├── __init__.py          # Package exports
│   ├── email_alert.py       # Email sending with HTML body and retry
│   ├── location.py          # IP-based location detection
│   ├── preprocessing.py     # CLAHE frame enhancement for fire visibility
│   └── file_utils.py        # Temp file cleanup
├── best.pt                  # Trained YOLOv11 fire segmentation model
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
├── .streamlit/
│   └── secrets.toml         # Email credentials (not tracked in git)
└── README.md                # This file
```

---

## Prerequisites

- Python 3.8 or higher
- Webcam (for live detection)
- Gmail account with [App Password](https://support.google.com/accounts/answer/185833) (for email alerts)
- Internet connection (for location services)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Avinash0377/AI-wildfire-alert-system.git
cd AI-wildfire-alert-system
```

### 2. Create Virtual Environment

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

### 4. Configure Email Credentials

Create `.streamlit/secrets.toml`:

```toml
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_google_app_password"
```

> **Note**: Generate a [Google App Password](https://support.google.com/accounts/answer/185833) for SMTP authentication. Never commit this file to git.

---

## Usage

### Run the Application

```bash
python -m streamlit run main.py
```

### Dashboard Controls

1. **Select Source** — Choose Webcam, RTSP/IP Camera, or Upload Video in the main area
2. **Start Detection** — Begin fire detection from the selected source
3. **Stop Detection** — Halt the detection process
4. **Sidebar Settings** — Configure receiver email, confidence threshold, and email cooldown

---

## Configuration

All configurable values are in `config.py`:

### Detection Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INFERENCE_IMG_SIZE` | 1280 | YOLO inference resolution (larger = more accurate) |
| `FRAME_WIDTH` | 1280 | Display frame width |
| `FRAME_HEIGHT` | 720 | Display frame height |
| `FRAME_SKIP` | 2 | Process every Nth frame |
| `DEFAULT_CONFIDENCE_THRESHOLD` | 50% | Minimum detection confidence |
| `CONSECUTIVE_FRAMES_REQUIRED` | 3 | Frames needed to confirm fire |
| `IOU_THRESHOLD` | 0.5 | Non-max suppression IOU threshold |

### Email Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_RECEIVER_EMAIL` | — | Alert recipient email |
| `DEFAULT_EMAIL_COOLDOWN` | 60s | Minimum seconds between email alerts |

### Preprocessing Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLAHE_CLIP_LIMIT` | 2.0 | Contrast enhancement strength |
| `CLAHE_TILE_SIZE` | (8, 8) | CLAHE grid size |
| `SATURATION_BOOST` | 1.2 | Color saturation multiplier |

---

## How It Works

1. **Video Input** — Accepts webcam, RTSP/IP camera stream, or uploaded video files
2. **Frame Preprocessing** — CLAHE contrast enhancement + saturation boost for fire visibility
3. **YOLO Inference** — YOLOv11-Seg detects and segments fire regions at 1280px resolution
4. **Temporal Confirmation** — Fire must appear in 3 consecutive frames to confirm detection
5. **Visualization** — Red segmentation masks and bounding boxes overlay on detected fire
6. **Email Alert** — HTML email with snapshot, location, timestamp, and confidence (60s cooldown)
7. **Logging** — All detections logged with class, confidence, bounding box, timestamp, and location

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not loading | Ensure `best.pt` is in the project root directory |
| Email not sending | Verify Gmail App Password in `.streamlit/secrets.toml` |
| Webcam not working | Check camera permissions; Windows uses `cv2.CAP_DSHOW` |
| Slow detection | Lower `INFERENCE_IMG_SIZE` in `config.py` or increase `FRAME_SKIP` |
| RTSP not connecting | Verify the RTSP URL format and network access |
| False positives | Increase `CONSECUTIVE_FRAMES_REQUIRED` or confidence threshold |

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Sudimalla Avinash**

- Email: [sudhimallaavinash@gmail.com](mailto:sudhimallaavinash@gmail.com)
- GitHub: [Avinash0377](https://github.com/Avinash0377)

---

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLOv11 model
- [Streamlit](https://streamlit.io/) for the dashboard framework
- [OpenCV](https://opencv.org/) for computer vision capabilities
