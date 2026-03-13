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

print("Loading face database upcoming...")





known_embeddings = []
known_names = []

for file in os.listdir(database_path):
    if file.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(database_path, file)

        embedding = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            enforce_detection=False
        )[0]["embedding"]

        known_embeddings.append(embedding)
        known_names.append(os.path.splitext(file)[0])

print("Database loaded successfully!")

# -----------------------------
# Setup MediaPipe
# -----------------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)

frame_count = 0
current_name = "Unknown"

# -----------------------------
# Real-time Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            face_img = frame[y:y+bh, x:x+bw]

            # Run recognition every 20 frames
            if frame_count % 20 == 0:
                try:
                    embedding = DeepFace.represent(
                        img_path=face_img,
                        model_name=model_name,
                        enforce_detection=False
                    )[0]["embedding"]

                    similarities = cosine_similarity(
                        [embedding],
                        known_embeddings
                    )[0]

                    max_index = np.argmax(similarities)
                    max_similarity = similarities[max_index]

                    if max_similarity > 0.6:   # threshold
                        current_name = known_names[max_index]
                    else:
                        current_name = "Unknown"

                except:
                    current_name = "Unknown"

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,255,0), 2)
            cv2.putText(frame, current_name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
