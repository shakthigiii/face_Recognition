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
frame_count = 0
current_name = "Scanning..." # Global status

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    # UI BUTTON LOGIC (Top Left Corner)
    # Background rectangle for the "Button"
    bg_color = (0, 255, 0) if current_name != "Unknown" and current_name != "Scanning..." else (0, 0, 255)
    if current_name == "Scanning...": bg_color = (255, 165, 0) # Orange for scanning
    
    cv2.rectangle(frame, (10, 10), (280, 60), bg_color, -1) # Solid Button
    cv2.rectangle(frame, (10, 10), (280, 60), (255, 255, 255), 2) # White Border
    
    status_text = f"STATUS: {current_name}"
    cv2.putText(frame, status_text, (25, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            x, y = max(0, x), max(0, y)

            # Recognition every 20 frames
            if frame_count % 20 == 0:
                face_img = frame[y:y+bh, x:x+bw]
                if face_img.size > 0:
                    try:
                        embedding = DeepFace.represent(img_path=face_img, model_name=model_name, enforce_detection=False)[0]["embedding"]
                        similarities = cosine_similarity([embedding], known_embeddings)[0]
                        max_idx = np.argmax(similarities)
                        
                        if similarities[max_idx] > 0.6:
                            current_name = known_names[max_idx].upper()
                        else:
                            current_name = "Unknown"
                    except:
                        current_name = "Unknown"

            # Draw face box (Matching the button color)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), bg_color, 2)

    cv2.imshow("Alzheimer's Assistant - Face ID", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()