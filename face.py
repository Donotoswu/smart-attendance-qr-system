import dlib
import cv2
import numpy as np
import os
import csv
from datetime import datetime, timedelta
import qrcode
import atexit
import pyttsx3
import getpass
import pickle
import logging
from scipy.spatial import KDTree
from playsound import playsound
import winsound
import geocoder

# Logging
logging.basicConfig(
    filename="attendance_system.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

# Dlib models
detector = dlib.get_frontal_face_detector()  # type: ignore
predictor = dlib.shape_predictor("dlib_models/shape_predictor_68_face_landmarks.dat")  # type: ignore
face_rec_model = dlib.face_recognition_model_v1("dlib_models//dlib_face_recognition_resnet_model_v1.dat")  # type: ignore

# Constants
attendance_log_file = "attendance_log.csv"
qr_generation_date_file = "qr_last_generated.txt"
WEB_CAM_PASSWORD = "admin123"
CLOSE_CAM_PASSWORD = "close123"
ATTENDANCE_COOLDOWN_MINUTES = 50
QR_CODE_COOLDOWN_DAYS = 182

# Globals
known_face_encodings = []
known_face_names = []
logged_names = set()
last_logged_time = {}
last_qr_message_shown = {}  # Initialize this variable
engine = pyttsx3.init()

# Load/Save Helpers
def load_logged_names():
    if os.path.exists("logged_names.txt"):
        with open("logged_names.txt", "r") as f:
            return set(f.read().splitlines())
    return set()

def save_logged_names():
    with open("logged_names.txt", "w") as f:
        f.write("\n".join(logged_names))

atexit.register(save_logged_names)

def load_qr_generation_dates():
    qr_dates = {}
    if os.path.exists(qr_generation_date_file):
        with open(qr_generation_date_file, "r", encoding="utf-8") as f:
            for line in f.read().splitlines():
                if ', ' in line:
                    parts = line.split(", ", 1)
                    if len(parts) == 2:
                        name, date_str = parts
                        try:
                            qr_dates[name.strip()] = datetime.strptime(date_str.strip(), "%Y-%m-%d %H:%M:%S.%f")
                        except ValueError:
                            logger.error(f"Failed to parse date for line: {line}")
    return qr_dates

def save_qr_generation_date(name, timestamp):
    with open(qr_generation_date_file, "a", encoding="utf-8") as f:
        f.write(f"{name}, {timestamp}\n")

# Face Recognition Helpers
def compare_faces_kdtree(face_encoding, tolerance=0.75):
    distance, index = face_tree.query(face_encoding)
    if distance < tolerance:
        return known_face_names[index], distance
    return None, distance

def save_known_faces():
    with open("known_faces.pkl", "wb") as f:
        pickle.dump(known_face_encodings, f)
        pickle.dump(known_face_names, f)
    logger.info("Known faces saved.")

def load_known_faces(base_dir):
    global known_face_encodings, known_face_names, face_tree

    if os.path.exists("known_faces.pkl"):
        with open("known_faces.pkl", "rb") as f:
            known_face_encodings = pickle.load(f)
            known_face_names = pickle.load(f)
        logger.info(f"Loaded {len(known_face_encodings)} known faces from pickle.")
    else:
        for person_name in os.listdir(base_dir):
            person_folder = os.path.join(base_dir, person_name)
            if not os.path.isdir(person_folder):
                continue
            for filename in os.listdir(person_folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(person_folder, filename)
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    faces = detector(image_rgb, 1)
                    for face in faces:
                        shape = predictor(image_rgb, face)
                        encoding = face_rec_model.compute_face_descriptor(image_rgb, shape)
                        if encoding:
                            known_face_encodings.append(np.array(encoding))
                            known_face_names.append(person_name)
        save_known_faces()

    face_tree = KDTree(known_face_encodings)

def can_generate_qr_code(name, last_generated_date):
    return (datetime.now() - last_generated_date) > timedelta(days=QR_CODE_COOLDOWN_DAYS) if last_generated_date else True

def can_log_attendance(name):
    now = datetime.now()
    last_time = last_logged_time.get(name)
    if last_time is None:
        last_logged_time[name] = now
        return True
    if now - last_time > timedelta(minutes=ATTENDANCE_COOLDOWN_MINUTES):
        last_logged_time[name] = now
        return True
    return False

# Attendance Logging
def log_attendance(name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 🗺 Get Location
    location = "Unknown"
    try:
        g = geocoder.ip('me')  # Get location based on IP
        location_info = g.json
        city = location_info.get("city", "Unknown City")
        street = location_info.get("street", "Unknown Street")
        state = location_info.get("state", "Unknown State")
        country = location_info.get("country", "Unknown Country")
        location = f"{city}, {street}, {state}, {country}"
    except Exception as e:
        logger.error(f"Location fetch failed: {e}")
        location = "Unknown"

    # Log to CSV
    file_exists = os.path.exists(attendance_log_file)
    with open(attendance_log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Timestamp", "Location"])
        writer.writerow([name, timestamp, location])

    print(f"✅ Attendance marked for {name} at {timestamp} in {location}")
    logger.info(f"Attendance marked for {name} at {timestamp} in {location}")

    # Sound Feedback
    try:
        playsound("attendance_marked.mp3")
    except Exception as e:
        logger.warning(f"Could not play attendance sound: {e}")

    try:
        winsound.Beep(1000, 600)
    except RuntimeError as e:
        logger.warning(f"Beep failed: {e}")

    # QR Code Handling
    qr_dates = load_qr_generation_dates()
    if name not in qr_dates or can_generate_qr_code(name, qr_dates.get(name)):
        qr_data = f"Attendance: {name}, Time: {timestamp}, Location: {location}"
        folder = "known_face_generated"
        os.makedirs(folder, exist_ok=True)
        img = qrcode.make(qr_data)
        img_path = os.path.join(folder, f"{name}_attendance.png")
        with open(img_path, 'wb') as f:
            img.save(f)
        save_qr_generation_date(name, datetime.now())
        print(f"🧾 QR code generated and saved for {name} at {img_path}")
        logger.info(f"QR code generated for {name} at {timestamp}")
    else:
        if not last_qr_message_shown.get(name):
            print(f"📛 QR code not generated for {name} (182-day limit active).")
            last_qr_message_shown[name] = True

# Password Handling
def request_password():
    return getpass.getpass("Enter webcam password: ") == WEB_CAM_PASSWORD

def request_close_password():
    return getpass.getpass("Enter password to close webcam: ") == CLOSE_CAM_PASSWORD

# QR Code Scanning
def scan_qr_code(frame):
    qr_decoder = cv2.QRCodeDetector()
    value, pts, _ = qr_decoder.detectAndDecode(frame)
    if value:
        print(f"QR Code detected: {value}")
        logger.info(f"QR Code detected: {value}")

# Main Recognition Loop
def recognize_faces_and_qr():
    if not request_password():
        logger.error("Incorrect webcam password. Exiting...")
        print("❌ Incorrect webcam password. Exiting...")
        return

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        scale = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_small_frame)

        scan_qr_code(frame)

        for face in faces:
            scaled_face = dlib.rectangle( # type: ignore
                int(face.left() / scale),
                int(face.top() / scale),
                int(face.right() / scale),
                int(face.bottom() / scale),
            )

            shape = predictor(rgb_small_frame, scaled_face)
            encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape))

            name = None
            label = "Unknown"

            name, distance = compare_faces_kdtree(encoding)

            if name and distance is not None:
                label = name
                if can_log_attendance(name):
                    log_attendance(name)
                    logged_names.add(name)
                    last_qr_message_shown[name] = False
                else:
                    print(f"⚠ Attendance already logged for {name} within the last {ATTENDANCE_COOLDOWN_MINUTES} minutes.")

            print(f"Recognized: {label} (distance: {distance})")

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load known faces
load_known_faces("known_faces")

# Start recognition
recognize_faces_and_qr()
