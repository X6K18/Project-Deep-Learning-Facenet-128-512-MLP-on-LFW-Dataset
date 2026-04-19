import cv2
import os
import pickle
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk
from deepface import DeepFace
from customtkinter import (
    CTkLabel, CTkScrollableFrame, CTkFrame, CTkButton,
    CTkImage, CTkEntry, CTkTextbox, CTkProgressBar
)

class FaceRegisterPage(CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs, fg_color="transparent")

        # -------------------- ICONS --------------------
        self.start_img = CTkImage(dark_image=Image.open("app/assets/img/start.png"), size=(45, 45))
        self.stop_img = CTkImage(dark_image=Image.open("app/assets/img/stop.png"), size=(45, 45))
        self.register_img = CTkImage(dark_image=Image.open("app/assets/img/face_registered.png"), size=(45, 45))

        # -------------------- CONFIG --------------------
        self.EMBEDDING_DIR = "app/data/embeddings"
        os.makedirs(self.EMBEDDING_DIR, exist_ok=True)
        self.MODEL_NAME = "Facenet512"
        self.DETECTOR_BACKEND = "opencv"

        # -------------------- CAMERA STATE --------------------
        self.is_running = False
        self.cap = None

        # -------------------- UI LAYOUT --------------------
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=1)

        # 1. WEBCAM CARD
        self.webcam_card = CTkFrame(self, corner_radius=15, border_width=2, border_color="gray40")
        self.webcam_card.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        self.webcam_card.grid_columnconfigure(0, weight=1)

        self.webcam_label = CTkLabel(
            self.webcam_card, text="Webcam is Offline",
            width=640, height=480, fg_color="gray20", corner_radius=10
        )
        self.webcam_label.grid(row=0, column=0, padx=10, pady=10)

        # 2. CONTROL BUTTONS
        self.controls_frame = CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=1, column=0, pady=10)

        self.start_btn = CTkButton(
            self.controls_frame, text="Start Webcam", font=("Times New Roman", 16, "bold"),
            border_width=3, image=self.start_img, hover_color="gray10",
            corner_radius=30, command=self.start_webcam
        )
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = CTkButton(
            self.controls_frame, text="Stop Webcam", font=("Times New Roman", 16, "bold"),
            border_width=3, image=self.stop_img, fg_color="transparent",
            corner_radius=30, command=self.stop_webcam
        )
        self.stop_btn.grid(row=0, column=1, padx=10)

        # 3. REGISTRATION FORM
        self.form_card = CTkFrame(self, corner_radius=15, border_width=2, border_color="gray40")
        self.form_card.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        self.form_card.grid_columnconfigure(0, weight=0)
        self.form_card.grid_columnconfigure(1, weight=1)

        CTkLabel(self.form_card, text="Register New Face", font=("Times New Roman", 18, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 5)
        )

        CTkLabel(self.form_card, text="Full Name:", font=("Times New Roman", 14)).grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )
        self.name_entry = CTkEntry(self.form_card, width=250)
        self.name_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        CTkLabel(self.form_card, text="Student ID:", font=("Times New Roman", 14)).grid(
            row=2, column=0, padx=10, pady=5, sticky="e"
        )
        self.id_entry = CTkEntry(self.form_card, width=250)
        self.id_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        self.register_btn = CTkButton(
            self.form_card, text="Register Face", font=("Times New Roman", 16, "bold"),
            image=self.register_img, compound="left", corner_radius=30,
            command=self.register_face, fg_color="#2C7A4D", hover_color="#1E5A38"
        )
        self.register_btn.grid(row=3, column=0, columnspan=2, pady=15)

        self.progress = CTkProgressBar(self.form_card, mode="indeterminate", height=8)
        self.progress.grid(row=4, column=0, columnspan=2, padx=20, pady=(0, 15), sticky="ew")
        self.progress.set(0)

        # 4. LOG AREA
        self.log_card = CTkFrame(self, corner_radius=15, border_width=2, border_color="gray40")
        self.log_card.grid(row=3, column=0, padx=10, pady=(5, 10), sticky="nsew")
        self.log_card.grid_columnconfigure(0, weight=1)
        self.log_card.grid_rowconfigure(1, weight=1)

        CTkLabel(self.log_card, text="Registration Log", font=("Times New Roman", 18, "bold")).grid(
            row=0, column=0, pady=(10, 5)
        )
        self.log_text = CTkTextbox(self.log_card, font=("Courier New", 12), wrap="word")
        self.log_text.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

        self.add_log("Ready. Enter information and start webcam.")
        self.bind("<Destroy>", self.on_close)

    # -------------------- WEBCAM METHODS --------------------
    def start_webcam(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.is_running = True
            self.webcam_label.configure(text="")
            self.update_webcam()
            self.add_log("Webcam started.")

    def stop_webcam(self):
        self.is_running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        # Chỉ cập nhật UI nếu widget vẫn còn tồn tại
        if self.webcam_label.winfo_exists():
            self.webcam_label.configure(image="", text="Webcam is Offline")
            self.webcam_label.image = None
        # Ghi log an toàn (kiểm tra log_text còn tồn tại)
        self.add_log("Webcam stopped.")

    def update_webcam(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.update_webcam)
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        size = min(w, h) // 2
        cv2.rectangle(frame, (center_x - size, center_y - size),
                      (center_x + size, center_y + size), (0, 255, 0), 2)
        cv2.putText(frame, "Place face inside the box", (center_x - 120, center_y - size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.webcam_label.configure(image=img_tk)
        self.webcam_label.image = img_tk
        self.after(10, self.update_webcam)

    # -------------------- LOGGING (AN TOÀN) --------------------
    def add_log(self, msg):
        """Chỉ ghi log nếu widget log_text vẫn còn tồn tại."""
        if not self.winfo_exists():
            return
        if hasattr(self, 'log_text') and self.log_text.winfo_exists():
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert("end", f"[{timestamp}] {msg}\n")
            self.log_text.see("end")

    # -------------------- REGISTRATION LOGIC --------------------
    def register_face(self):
        if not self.is_running or self.cap is None:
            self.add_log("ERROR: Webcam is not running. Please start the webcam first.")
            return

        name = self.name_entry.get().strip()
        student_id = self.id_entry.get().strip()
        if not name or not student_id:
            self.add_log("ERROR: Please enter both Full Name and Student ID.")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.add_log("ERROR: Failed to capture frame.")
            return

        frame = cv2.flip(frame, 1)
        self.progress.start()
        self.add_log(f"Processing face for {name} (ID: {student_id})...")

        try:
            embedding_objs = DeepFace.represent(
                img_path=frame,
                model_name=self.MODEL_NAME,
                enforce_detection=True,
                detector_backend=self.DETECTOR_BACKEND
            )
            if not embedding_objs:
                self.add_log("ERROR: No face detected. Please look straight into the camera.")
                self.progress.stop()
                self.progress.set(0)
                return

            embedding = embedding_objs[0]["embedding"]
            facial_area = embedding_objs[0]["facial_area"]

            data = {
                "name": name,
                "student_id": student_id,
                "embedding": np.array(embedding),
                "timestamp": datetime.now().isoformat(),
                "facial_area": facial_area
            }
            filename = f"{student_id}_{name.replace(' ', '_')}.pkl"
            filepath = os.path.join(self.EMBEDDING_DIR, filename)
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

            self.add_log(f"SUCCESS: Registered {name} (ID: {student_id})")
            self.add_log(f"Embedding saved to: {filepath}")
            self.progress.stop()
            self.progress.set(1)
        except Exception as e:
            self.add_log(f"ERROR: {str(e)}")
            self.progress.stop()
            self.progress.set(0)

    def on_close(self, event=None):
        self.stop_webcam()