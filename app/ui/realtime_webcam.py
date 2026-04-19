# import cv2
# import numpy as np
# import pickle
# from deepface import DeepFace
# from tensorflow.keras.models import load_model

# # ===== Load =====
# model = load_model("face_mlp.h5")

# with open("norm.pkl", "rb") as f:
#     norm = pickle.load(f)

# with open("label_map.pkl", "rb") as f:
#     label_map = pickle.load(f)

# # ===== Webcam =====
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()

#     try:
#         # detect + embedding
#         reps = DeepFace.represent(
#             frame,
#             model_name="Facenet512",
#             enforce_detection=False
#         )

#         emb = np.array(reps[0]["embedding"]).reshape(1, -1)
#         emb = norm.transform(emb)

#         pred = model.predict(emb)
#         label = np.argmax(pred)
#         name = label_map[label]

#         cv2.putText(frame, name, (50,50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#     except:
#         pass

#     cv2.imshow("Face Recognition", frame)

#     if cv2.waitKey(1) == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import pickle
from deepface import DeepFace
from tensorflow.keras.models import load_model

# ===== Load model =====
model = load_model("face_mlp.h5")

with open("norm.pkl", "rb") as f:
    norm = pickle.load(f)

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# ===== Webcam =====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # ===== Detect face =====
        detections = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="opencv",
            enforce_detection=False
        )

        for face in detections:
            x, y, w, h = face["facial_area"].values()

            # ===== Crop face =====
            face_img = frame[y:y+h, x:x+w]

            # ===== Embedding =====
            rep = DeepFace.represent(
                face_img,
                model_name="Facenet512",
                enforce_detection=False
            )

            emb = np.array(rep[0]["embedding"]).reshape(1, -1)
            emb = norm.transform(emb)

            # ===== Predict =====
            pred = model.predict(emb)
            prob = np.max(pred)
            label = np.argmax(pred)

            name = label_map[label]

            # ===== Threshold Unknown =====
            if prob < 0.7:
                name = "Unknown"

            # ===== Draw bounding box =====
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # ===== Show name + confidence =====
            text = f"{name} ({prob:.2f})"
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

    except Exception as e:
        pass

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()