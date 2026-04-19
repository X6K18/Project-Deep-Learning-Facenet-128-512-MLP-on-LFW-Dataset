import customtkinter as ctk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import pickle
from deepface import DeepFace
from tensorflow.keras.models import load_model

# ===== Load model =====
model = load_model("face_mlp.h5")

with open("norm.pkl", "rb") as f:
    norm = pickle.load(f)

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# ===== App config =====
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Face Recognition System")
app.geometry("1000x700")

# ===== Variables =====
img1_path = None
img2_path = None
cap = None
is_running = False

# ===== Functions =====
def show_image(path, label):
    img = Image.open(path)
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)
    label.configure(image=img)
    label.image = img

def load_image_1():
    global img1_path
    img1_path = filedialog.askopenfilename()
    if img1_path:
        show_image(img1_path, label_img1)

def load_image_2():
    global img2_path
    img2_path = filedialog.askopenfilename()
    if img2_path:
        show_image(img2_path, label_img2)

def get_embedding(img):
    rep = DeepFace.represent(
        img,
        model_name="Facenet512",
        enforce_detection=False
    )
    emb = np.array(rep[0]["embedding"]).reshape(1, -1)
    emb = norm.transform(emb)
    return emb

# ===== Compare 2 images =====
def compare_faces():
    if not img1_path or not img2_path:
        result_label.configure(text="⚠️ Chọn đủ 2 ảnh!")
        return

    try:
        emb1 = get_embedding(img1_path)
        emb2 = get_embedding(img2_path)

        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(emb1, emb2)[0][0]

        if sim > 0.7:
            result = "✅ SAME PERSON"
            color = "green"
        else:
            result = "❌ DIFFERENT PERSON"
            color = "red"

        result_label.configure(
            text=f"{result}\nSimilarity: {sim:.2f}",
            text_color=color
        )

    except Exception as e:
        result_label.configure(text=f"Lỗi: {str(e)}", text_color="red")

# ===== Webcam =====
def start_webcam():
    global cap, is_running
    cap = cv2.VideoCapture(0)
    is_running = True
    update_frame()

def stop_webcam():
    global is_running, cap
    is_running = False
    if cap:
        cap.release()

def update_frame():
    if not is_running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.resize(frame, (640, 400))

    try:
        detections = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="opencv",
            enforce_detection=False
        )

        for face in detections:
            x, y, w, h = face["facial_area"].values()

            face_img = frame[y:y+h, x:x+w]

            emb = get_embedding(face_img)

            pred = model.predict(emb)
            prob = np.max(pred)
            label = np.argmax(pred)

            name = label_map[label]

            if prob < 0.7:
                name = "Unknown"

            color = (0,255,0) if name != "Unknown" else (0,0,255)

            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({prob:.2f})",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

    except:
        pass

    # Convert to Tkinter image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    webcam_label.imgtk = imgtk
    webcam_label.configure(image=imgtk)

    webcam_label.after(10, update_frame)

# ===== UI Layout =====
top_frame = ctk.CTkFrame(app)
top_frame.pack(pady=10)

# Image 1
btn1 = ctk.CTkButton(top_frame, text="Chọn ảnh 1", command=load_image_1)
btn1.grid(row=0, column=0, padx=20)

label_img1 = ctk.CTkLabel(top_frame, text="Ảnh 1", width=200, height=200)
label_img1.grid(row=1, column=0)

# Image 2
btn2 = ctk.CTkButton(top_frame, text="Chọn ảnh 2", command=load_image_2)
btn2.grid(row=0, column=1, padx=20)

label_img2 = ctk.CTkLabel(top_frame, text="Ảnh 2", width=200, height=200)
label_img2.grid(row=1, column=1)

# Compare button
compare_btn = ctk.CTkButton(app, text="So sánh 2 ảnh", command=compare_faces)
compare_btn.pack(pady=10)

# Result
result_label = ctk.CTkLabel(app, text="", font=("Arial", 16))
result_label.pack()

# Webcam controls
webcam_frame = ctk.CTkFrame(app)
webcam_frame.pack(pady=20)

start_btn = ctk.CTkButton(webcam_frame, text="Start Webcam", command=start_webcam)
start_btn.grid(row=0, column=0, padx=10)

stop_btn = ctk.CTkButton(webcam_frame, text="Stop Webcam", command=stop_webcam)
stop_btn.grid(row=0, column=1, padx=10)

webcam_label = ctk.CTkLabel(app, text="")
webcam_label.pack()

app.mainloop()