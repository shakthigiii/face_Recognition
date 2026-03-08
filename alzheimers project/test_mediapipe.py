import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Face Database ONCE
# -----------------------------
database_path = "database"
model_name = "Facenet"

print("Loading face database...")
known_embeddings = []
known_names = []

for file in os.listdir(database_path):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(database_path, file)
        try:
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                enforce_detection=False
            )[0]["embedding"]
            known_embeddings.append(embedding)
            known_names.append(os.path.splitext(file)[0])
        except Exception as e:
            print(f"Error loading {file}: {e}")

print(f"Database loaded! {len(known_names)} faces indexed.")

# -----------------------------
# Setup MediaPipe
# -----------------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)
frame_count = 1
current_name = "Scanning..." # Global variable to store last recognized name

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            # Get Bounding Box
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Keep box inside frame boundaries
            x, y = max(0, x), max(0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)

            # --- Recognition Logic (Throttled) ---
            if frame_count % 20 == 0:
                face_img = frame[y:y+bh, x:x+bw]
                if face_img.size > 0:
                    try:
                        embedding = DeepFace.represent(
                            img_path=face_img,
                            model_name=model_name,
                            enforce_detection=False
                        )[0]["embedding"]

                        similarities = cosine_similarity([embedding], known_embeddings)[0]
                        max_idx = np.argmax(similarities)
                        
                        if similarities[max_idx] > 0.6: # Threshold
                            current_name = known_names[max_idx]
                        else:
                            current_name = "Unknown"
                    except:
                        current_name = "Unknown"

            # --- Drawing Logic ---
            # Define color: Green for known, Red for unknown
            color = (0, 255, 0) if current_name != "Unknown" else (0, 0, 255)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
            
            # Place text EXACTLY above the rectangle
            # If y-10 is off-screen, it puts it at y+bh+20
            label_y = y - 10 if y - 10 > 20 else y + bh + 20
            cv2.putText(frame, current_name, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()