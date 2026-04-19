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

# ===== App config =====
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Face Verification")
app.geometry("800x500")

# ===== Variables =====
img1_path = None
img2_path = None

# ===== Functions =====
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

def show_image(path, label):
    img = Image.open(path)
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)
    label.configure(image=img)
    label.image = img

def get_embedding(img_path):
    rep = DeepFace.represent(
        img_path=img_path,
        model_name="Facenet512",
        enforce_detection=False
    )
    emb = np.array(rep[0]["embedding"]).reshape(1, -1)
    emb = norm.transform(emb)
    return emb

def compare_faces():
    if not img1_path or not img2_path:
        result_label.configure(text="⚠️ Chọn đủ 2 ảnh!")
        return

    try:
        emb1 = get_embedding(img1_path)
        emb2 = get_embedding(img2_path)

        # ===== Predict =====
        pred1 = model.predict(emb1)
        pred2 = model.predict(emb2)

        label1 = np.argmax(pred1)
        label2 = np.argmax(pred2)

        conf1 = np.max(pred1)
        conf2 = np.max(pred2)

        # ===== Compare =====
        if label1 == label2:
            result = "✅ SAME PERSON"
            color = "green"
        else:
            result = "❌ DIFFERENT PERSON"
            color = "red"

        result_label.configure(
            text=f"{result}\nConf1: {conf1:.2f} | Conf2: {conf2:.2f}",
            text_color=color
        )

    except Exception as e:
        result_label.configure(text=f"Lỗi: {str(e)}", text_color="red")

# ===== UI Layout =====
frame = ctk.CTkFrame(app)
frame.pack(pady=20)

# Image 1
btn1 = ctk.CTkButton(frame, text="Chọn ảnh 1", command=load_image_1)
btn1.grid(row=0, column=0, padx=20)

label_img1 = ctk.CTkLabel(frame, text="Ảnh 1", width=200, height=200)
label_img1.grid(row=1, column=0, padx=20)

# Image 2
btn2 = ctk.CTkButton(frame, text="Chọn ảnh 2", command=load_image_2)
btn2.grid(row=0, column=1, padx=20)

label_img2 = ctk.CTkLabel(frame, text="Ảnh 2", width=200, height=200)
label_img2.grid(row=1, column=1, padx=20)

# Compare button
compare_btn = ctk.CTkButton(app, text="So sánh", command=compare_faces)
compare_btn.pack(pady=20)

# Result
result_label = ctk.CTkLabel(app, text="", font=("Arial", 16))
result_label.pack()

app.mainloop()