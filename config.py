# ======================== APP CONFIGURATION ========================
# Default values and constants for the Fire Detection System

# YOLO Model
MODEL_PATH = "best.pt"
INFERENCE_IMG_SIZE = 1280
IOU_THRESHOLD = 0.5

# Frame Processing
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_SKIP = 2  # Process every Nth frame

# Email Defaults
DEFAULT_RECEIVER_EMAIL = "sudhimallaavinash03@gmail.com"
DEFAULT_EMAIL_COOLDOWN = 60  # seconds

# Detection Defaults
DEFAULT_CONFIDENCE_THRESHOLD = 50  # percent
CONSECUTIVE_FRAMES_REQUIRED = 3  # frames of fire before confirming

# CLAHE Preprocessing
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
SATURATION_BOOST = 1.2

# Session State Defaults
SESSION_DEFAULTS = {
    'detection_running': False,
    'video_source': None,
    'video_source_type': None,
    'email_threads': [],
    'live_location': None,
    'fire_count': 0,
    'detection_log': [],
    'last_email_time': 0,
    'temp_file_path': None,
    'fire_consecutive_frames': 0,
}
