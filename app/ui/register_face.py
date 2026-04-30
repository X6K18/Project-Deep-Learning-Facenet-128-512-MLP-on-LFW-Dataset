# import cv2
# import os
# import pickle
# import numpy as np
# from datetime import datetime
# from PIL import Image, ImageTk
# from deepface import DeepFace
# from customtkinter import (
#     CTkLabel, CTkScrollableFrame, CTkFrame, CTkButton,
#     CTkImage, CTkEntry, CTkTextbox, CTkProgressBar, CTkSwitch,
#     CTkOptionMenu
# )

# class FaceRegisterPage(CTkScrollableFrame):
#     def __init__(self, master, **kwargs):
#         super().__init__(master, **kwargs, fg_color="transparent")

#         # -------------------- ICONS --------------------
#         self.start_img = CTkImage(dark_image=Image.open("app/assets/img/start.png"), size=(45, 45))
#         self.stop_img = CTkImage(dark_image=Image.open("app/assets/img/stop.png"), size=(45, 45))
#         self.register_img = CTkImage(dark_image=Image.open("app/assets/img/face_registered.png"), size=(45, 45))
#         self.capture_img = CTkImage(dark_image=Image.open("app/assets/img/camera.png"), size=(45, 45))

#         # -------------------- CONFIG --------------------
#         self.EMBEDDING_DIR = "app/data/embeddings"
#         os.makedirs(self.EMBEDDING_DIR, exist_ok=True)
#         self.MODEL_NAME = "Facenet512"
#         self.DETECTOR_BACKEND = "opencv"
        
#         # Cài đặt mặc định
#         self.NUM_SAMPLES = 5               # số mẫu cần chụp
#         self.AUTO_DELAY_SEC = 1.5          # giây giữa các lần auto capture
#         self.auto_capture_enabled = False  # trạng thái auto
#         self.last_auto_capture_time = 0
#         self.stable_face_counter = 0       # đếm số frame có khuôn mặt ổn định
        
#         # Dữ liệu thu thập
#         self.embeddings_samples = []
        
#         # -------------------- CAMERA STATE --------------------
#         self.is_running = False
#         self.cap = None

#         # -------------------- UI LAYOUT --------------------
#         self.grid_columnconfigure(0, weight=1)
#         self.grid_rowconfigure(0, weight=0)
#         self.grid_rowconfigure(1, weight=0)
#         self.grid_rowconfigure(2, weight=0)
#         self.grid_rowconfigure(3, weight=1)

#         # 1. WEBCAM CARD
#         self.webcam_card = CTkFrame(self, corner_radius=15, border_width=2, border_color="gray40")
#         self.webcam_card.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
#         self.webcam_card.grid_columnconfigure(0, weight=1)

#         self.webcam_label = CTkLabel(
#             self.webcam_card, text="Webcam is Offline",
#             width=640, height=480, fg_color="gray20", corner_radius=10
#         )
#         self.webcam_label.grid(row=0, column=0, padx=10, pady=10)

#         # 2. CONTROL BUTTONS
#         self.controls_frame = CTkFrame(self, fg_color="transparent")
#         self.controls_frame.grid(row=1, column=0, pady=10)

#         self.start_btn = CTkButton(
#             self.controls_frame, text="Start Webcam", font=("Times New Roman", 16, "bold"),
#             border_width=3, image=self.start_img, hover_color="gray10",
#             corner_radius=30, command=self.start_webcam
#         )
#         self.start_btn.grid(row=0, column=0, padx=10)

#         self.stop_btn = CTkButton(
#             self.controls_frame, text="Stop Webcam", font=("Times New Roman", 16, "bold"),
#             border_width=3, image=self.stop_img, fg_color="transparent",
#             corner_radius=30, command=self.stop_webcam
#         )
#         self.stop_btn.grid(row=0, column=1, padx=10)

#         # 3. REGISTRATION FORM + CAPTURE AREA
#         self.form_card = CTkFrame(self, corner_radius=15, border_width=2, border_color="gray40")
#         self.form_card.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
#         self.form_card.grid_columnconfigure(0, weight=0)
#         self.form_card.grid_columnconfigure(1, weight=1)
#         self.form_card.grid_columnconfigure(2, weight=0)

#         CTkLabel(self.form_card, text="Register New Face", font=("Times New Roman", 18, "bold")).grid(
#             row=0, column=0, columnspan=4, pady=(10, 5)
#         )

#         # Hàng 1: Họ tên
#         CTkLabel(self.form_card, text="Full Name:", font=("Times New Roman", 14)).grid(
#             row=1, column=0, padx=10, pady=5, sticky="e"
#         )
#         self.name_entry = CTkEntry(self.form_card, width=250)
#         self.name_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w", columnspan=2)

#         # Hàng 2: Mã số sinh viên
#         CTkLabel(self.form_card, text="Student ID:", font=("Times New Roman", 14)).grid(
#             row=2, column=0, padx=10, pady=5, sticky="e"
#         )
#         self.id_entry = CTkEntry(self.form_card, width=250)
#         self.id_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w", columnspan=2)

#         # Hàng 3: Auto Capture Switch + Số mẫu + Delay
#         self.auto_capture_switch = CTkSwitch(
#             self.form_card, text="Auto Capture", command=self.toggle_auto_capture,
#             font=("Times New Roman", 14)
#         )
#         self.auto_capture_switch.grid(row=3, column=0, padx=10, pady=5, sticky="w")

#         CTkLabel(self.form_card, text="Samples:", font=("Times New Roman", 12)).grid(
#             row=3, column=1, padx=(5,0), pady=5, sticky="e"
#         )
#         self.samples_option = CTkOptionMenu(
#             self.form_card, values=["3", "5", "7", "10"], width=60,
#             command=self.change_num_samples
#         )
#         self.samples_option.set(str(self.NUM_SAMPLES))
#         self.samples_option.grid(row=3, column=1, padx=(50,5), pady=5, sticky="w")

#         CTkLabel(self.form_card, text="Delay (s):", font=("Times New Roman", 12)).grid(
#             row=3, column=2, padx=(5,0), pady=5, sticky="e"
#         )
#         self.delay_option = CTkOptionMenu(
#             self.form_card, values=["1.0", "1.5", "2.0", "2.5"], width=60,
#             command=self.change_auto_delay
#         )
#         self.delay_option.set(str(self.AUTO_DELAY_SEC))
#         self.delay_option.grid(row=3, column=2, padx=(50,5), pady=5, sticky="w")

#         # Hàng 4: Nút Capture thủ công và Save
#         self.capture_btn = CTkButton(
#             self.form_card, text="Capture Face", font=("Times New Roman", 14, "bold"),
#             image=self.capture_img, compound="left", corner_radius=30,
#             command=self.capture_sample, fg_color="#2C7A4D", hover_color="#1E5A38"
#         )
#         self.capture_btn.grid(row=4, column=0, padx=10, pady=10)

#         self.register_btn = CTkButton(
#             self.form_card, text="Save Registration", font=("Times New Roman", 14, "bold"),
#             image=self.register_img, compound="left", corner_radius=30,
#             command=self.register_face, fg_color="#2C7A4D", hover_color="#1E5A38"
#         )
#         self.register_btn.grid(row=4, column=1, padx=10, pady=10)

#         self.sample_count_label = CTkLabel(self.form_card, text=f"Samples: 0/{self.NUM_SAMPLES}", font=("Times New Roman", 12))
#         self.sample_count_label.grid(row=4, column=2, padx=10, pady=10)

#         # Preview ảnh vừa chụp
#         self.preview_label = CTkLabel(
#             self.form_card, text="Preview", width=120, height=120,
#             fg_color="gray20", corner_radius=10
#         )
#         self.preview_label.grid(row=5, column=0, columnspan=3, padx=10, pady=5)

#         self.progress = CTkProgressBar(self.form_card, mode="indeterminate", height=8)
#         self.progress.grid(row=6, column=0, columnspan=3, padx=20, pady=(0, 15), sticky="ew")
#         self.progress.set(0)

#         # 4. LOG AREA
#         self.log_card = CTkFrame(self, corner_radius=15, border_width=2, border_color="gray40")
#         self.log_card.grid(row=3, column=0, padx=10, pady=(5, 10), sticky="nsew")
#         self.log_card.grid_columnconfigure(0, weight=1)
#         self.log_card.grid_rowconfigure(1, weight=1)

#         CTkLabel(self.log_card, text="Registration Log", font=("Times New Roman", 18, "bold")).grid(
#             row=0, column=0, pady=(10, 5)
#         )
#         self.log_text = CTkTextbox(self.log_card, font=("Courier New", 12), wrap="word")
#         self.log_text.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

#         self.add_log("Ready. Start webcam, fill info. Enable Auto Capture or use manual Capture.")
#         self.bind("<Destroy>", self.on_close)

#     # -------------------- WEBCAM METHODS --------------------
#     def start_webcam(self):
#         if not self.is_running:
#             self.cap = cv2.VideoCapture(0)
#             self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#             self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#             self.is_running = True
#             self.webcam_label.configure(text="")
#             self.update_webcam()
#             self.add_log("Webcam started.")

#     def stop_webcam(self):
#         self.is_running = False
#         if self.cap is not None and self.cap.isOpened():
#             self.cap.release()
#             self.cap = None
#         if self.webcam_label.winfo_exists():
#             self.webcam_label.configure(image="", text="Webcam is Offline")
#             self.webcam_label.image = None
#         self.add_log("Webcam stopped.")

#     def update_webcam(self):
#         if not self.is_running or self.cap is None:
#             return

#         ret, frame = self.cap.read()
#         if not ret:
#             self.after(10, self.update_webcam)
#             return

#         frame = cv2.flip(frame, 1)
#         h, w = frame.shape[:2]
#         center_x, center_y = w // 2, h // 2
#         size = min(w, h) // 2
#         cv2.rectangle(frame, (center_x - size, center_y - size),
#                       (center_x + size, center_y + size), (0, 255, 0), 2)
#         cv2.putText(frame, "Place face inside the box", (center_x - 120, center_y - size - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#         # Auto capture logic
#         if self.auto_capture_enabled and len(self.embeddings_samples) < self.NUM_SAMPLES:
#             # Gọi hàm xử lý auto capture
#             self.process_auto_capture(frame)

#         # Chuyển đổi để hiển thị
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img_pil = Image.fromarray(frame_rgb)
#         img_tk = ImageTk.PhotoImage(image=img_pil)
#         self.webcam_label.configure(image=img_tk)
#         self.webcam_label.image = img_tk
#         self.after(10, self.update_webcam)

#     # -------------------- AUTO CAPTURE LOGIC --------------------
#     def process_auto_capture(self, frame):
#         """Kiểm tra điều kiện và tự động chụp nếu đủ ổn định"""
#         now = datetime.now().timestamp()
#         # Nếu chưa đủ thời gian delay kể từ lần capture cuối thì bỏ qua
#         if now - self.last_auto_capture_time < self.AUTO_DELAY_SEC:
#             return

#         # Phát hiện khuôn mặt
#         try:
#             faces = DeepFace.extract_faces(
#                 img_path=frame,
#                 detector_backend=self.DETECTOR_BACKEND,
#                 enforce_detection=False
#             )
#         except Exception:
#             self.stable_face_counter = 0
#             return

#         if not faces or len(faces) == 0:
#             self.stable_face_counter = 0
#             return

#         # Lấy khuôn mặt đầu tiên
#         face = faces[0]
#         facial_area = face["facial_area"]
#         # Kiểm tra kích thước khuôn mặt (đủ lớn)
#         face_w = facial_area["w"]
#         face_h = facial_area["h"]
#         frame_h, frame_w = frame.shape[:2]
#         if face_w < frame_w * 0.2 or face_h < frame_h * 0.2:
#             # Khuôn mặt quá nhỏ
#             self.stable_face_counter = 0
#             return

#         # Kiểm tra độ ổn định: khuôn mặt nằm gần trung tâm
#         center_x, center_y = frame_w // 2, frame_h // 2
#         face_center_x = facial_area["x"] + face_w // 2
#         face_center_y = facial_area["y"] + face_h // 2
#         offset = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
#         max_offset = min(frame_w, frame_h) * 0.15
#         if offset > max_offset:
#             self.stable_face_counter = 0
#             return

#         # Tăng bộ đếm ổn định
#         self.stable_face_counter += 1
#         if self.stable_face_counter >= 5:  # 5 frame liên tiếp ổn định (khoảng 0.5s)
#             # Thực hiện auto capture
#             self.stable_face_counter = 0
#             self.last_auto_capture_time = now
#             # Gọi capture_sample với frame hiện tại
#             self.capture_sample(frame)

#     def toggle_auto_capture(self):
#         self.auto_capture_enabled = self.auto_capture_switch.get() == 1
#         if self.auto_capture_enabled:
#             self.add_log("Auto Capture enabled. Keep your face stable in the center.")
#             self.stable_face_counter = 0
#             self.last_auto_capture_time = 0
#         else:
#             self.add_log("Auto Capture disabled.")

#     def change_num_samples(self, value):
#         self.NUM_SAMPLES = int(value)
#         self.sample_count_label.configure(text=f"Samples: {len(self.embeddings_samples)}/{self.NUM_SAMPLES}")
#         self.add_log(f"Number of samples changed to {self.NUM_SAMPLES}")

#     def change_auto_delay(self, value):
#         self.AUTO_DELAY_SEC = float(value)
#         self.add_log(f"Auto capture delay set to {self.AUTO_DELAY_SEC} seconds")

#     # -------------------- CAPTURE & REGISTRATION --------------------
#     def capture_sample(self, frame=None):
#         """
#         Chụp một mẫu khuôn mặt.
#         Nếu frame được truyền vào (auto capture), dùng frame đó; nếu không thì lấy từ webcam.
#         """
#         if not self.is_running or self.cap is None:
#             self.add_log("ERROR: Webcam is not running. Please start webcam first.")
#             return

#         if len(self.embeddings_samples) >= self.NUM_SAMPLES:
#             self.add_log(f"Already captured {self.NUM_SAMPLES} samples. Press 'Save Registration'.")
#             return

#         if frame is None:
#             ret, frame = self.cap.read()
#             if not ret:
#                 self.add_log("ERROR: Failed to capture frame.")
#                 return
#             frame = cv2.flip(frame, 1)
#         else:
#             # Nếu frame từ auto capture, nó đã được flip chưa? Trong process_auto_capture, frame gốc chưa flip.
#             # Nhưng trong update_webcam, frame đã được flip. Để đồng nhất, ta flip lại nếu cần.
#             # Thực tế, frame từ update_webcam đã flip, không cần flip thêm.
#             pass

#         self.progress.start()
#         self.add_log(f"Capturing sample {len(self.embeddings_samples)+1}/{self.NUM_SAMPLES}...")

#         try:
#             embedding_objs = DeepFace.represent(
#                 img_path=frame,
#                 model_name=self.MODEL_NAME,
#                 enforce_detection=True,
#                 detector_backend=self.DETECTOR_BACKEND
#             )
#             if not embedding_objs:
#                 self.add_log("No face detected. Please look straight into the camera.")
#                 self.progress.stop()
#                 return

#             embedding = embedding_objs[0]["embedding"]
#             facial_area = embedding_objs[0]["facial_area"]

#             self.embeddings_samples.append(np.array(embedding))

#             # Cắt vùng mặt để preview
#             x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
#             face_roi = frame[y:y+h, x:x+w]
#             if face_roi.size > 0:
#                 face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
#                 face_pil.thumbnail((120, 120), Image.Resampling.LANCZOS)
#                 preview_img = CTkImage(light_image=face_pil, dark_image=face_pil, size=(face_pil.width, face_pil.height))
#                 self.preview_label.configure(image=preview_img, text="")
#                 self.preview_label.image = preview_img

#             self.add_log(f"Sample {len(self.embeddings_samples)} captured successfully.")
#             self.sample_count_label.configure(text=f"Samples: {len(self.embeddings_samples)}/{self.NUM_SAMPLES}")

#             if len(self.embeddings_samples) == self.NUM_SAMPLES:
#                 self.add_log("All samples captured. You can now save registration.")
#         except Exception as e:
#             self.add_log(f"Capture error: {str(e)}")
#         finally:
#             self.progress.stop()
#             self.progress.set(0)

#     def register_face(self):
#         """Tính embedding trung bình từ các mẫu và lưu vào file"""
#         name = self.name_entry.get().strip()
#         student_id = self.id_entry.get().strip()
#         if not name or not student_id:
#             self.add_log("ERROR: Please enter both Full Name and Student ID.")
#             return

#         if len(self.embeddings_samples) == 0:
#             self.add_log("ERROR: No face samples captured. Please capture at least one sample.")
#             return

#         avg_embedding = np.mean(self.embeddings_samples, axis=0)
#         data = {
#             "name": name,
#             "student_id": student_id,
#             "embedding": avg_embedding,
#             "timestamp": datetime.now().isoformat(),
#             "num_samples": len(self.embeddings_samples)
#         }
#         filename = f"{student_id}_{name.replace(' ', '_')}.pkl"
#         filepath = os.path.join(self.EMBEDDING_DIR, filename)
#         with open(filepath, "wb") as f:
#             pickle.dump(data, f)

#         self.add_log(f"SUCCESS: Registered {name} (ID: {student_id}) with {len(self.embeddings_samples)} samples.")
#         self.add_log(f"Embedding saved to: {filepath}")

#         self.reset_registration()

#     def reset_registration(self):
#         """Xoá dữ liệu mẫu, reset form"""
#         self.embeddings_samples = []
#         self.sample_count_label.configure(text=f"Samples: 0/{self.NUM_SAMPLES}")
#         self.preview_label.configure(image="", text="Preview")
#         self.name_entry.delete(0, "end")
#         self.id_entry.delete(0, "end")
#         self.add_log("Ready for next registration.")

#     # -------------------- LOGGING --------------------
#     def add_log(self, msg):
#         if not self.winfo_exists():
#             return
#         if hasattr(self, 'log_text') and self.log_text.winfo_exists():
#             timestamp = datetime.now().strftime("%H:%M:%S")
#             self.log_text.insert("end", f"[{timestamp}] {msg}\n")
#             self.log_text.see("end")

#     def on_close(self, event=None):
#         self.stop_webcam()



import cv2
import os
import pickle
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk
from deepface import DeepFace
from customtkinter import (
    CTkLabel, CTkScrollableFrame, CTkFrame, CTkButton,
    CTkImage, CTkEntry, CTkTextbox, CTkProgressBar, CTkSwitch,
    CTkOptionMenu
)

class FaceRegisterPage(CTkScrollableFrame):
    """Trang đăng ký khuôn mặt – lưu embedding chuẩn hóa để nhận diện sau."""

    # -------------------- PHƯƠNG THỨC TĨNH HỖ TRỢ NHẬN DIỆN --------------------
    @staticmethod
    def load_all_embeddings(embedding_dir="app/data/embeddings"):
        """
        Load tất cả các embedding đã đăng ký.
        Trả về danh sách các dict: {"name": str, "student_id": str, "embedding": np.ndarray}
        """
        registered = []
        if not os.path.exists(embedding_dir):
            return registered
        for fname in os.listdir(embedding_dir):
            if fname.endswith(".pkl"):
                filepath = os.path.join(embedding_dir, fname)
                try:
                    with open(filepath, "rb") as f:
                        data = pickle.load(f)
                    # Đảm bảo embedding là numpy array
                    emb = data.get("embedding")
                    if emb is not None:
                        if isinstance(emb, list):
                            emb = np.array(emb)
                        registered.append({
                            "name": data["name"],
                            "student_id": data["student_id"],
                            "embedding": emb
                        })
                except Exception as e:
                    print(f"Lỗi load {filepath}: {e}")
        return registered

    # -------------------- KHỞI TẠO --------------------
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs, fg_color="transparent")

        # -------------------- ICONS --------------------
        self.start_img = CTkImage(dark_image=Image.open("app/assets/img/start.png"), size=(45, 45))
        self.stop_img = CTkImage(dark_image=Image.open("app/assets/img/stop.png"), size=(45, 45))
        self.register_img = CTkImage(dark_image=Image.open("app/assets/img/face_registered.png"), size=(45, 45))
        self.capture_img = CTkImage(dark_image=Image.open("app/assets/img/camera.png"), size=(45, 45))

        # -------------------- CONFIG --------------------
        self.EMBEDDING_DIR = "app/data/embeddings"
        os.makedirs(self.EMBEDDING_DIR, exist_ok=True)
        self.MODEL_NAME = "Facenet512"
        self.DETECTOR_BACKEND = "opencv"

        # Cài đặt mặc định
        self.NUM_SAMPLES = 5               # số mẫu cần chụp
        self.AUTO_DELAY_SEC = 1.5          # giây giữa các lần auto capture
        self.auto_capture_enabled = False  # trạng thái auto
        self.last_auto_capture_time = 0
        self.stable_face_counter = 0       # đếm số frame có khuôn mặt ổn định

        # Dữ liệu thu thập tạm thời
        self.embeddings_samples = []       # lưu các embedding (np.array)

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

        # 3. REGISTRATION FORM + CAPTURE AREA
        self.form_card = CTkFrame(self, corner_radius=15, border_width=2, border_color="gray40")
        self.form_card.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        self.form_card.grid_columnconfigure(0, weight=0)
        self.form_card.grid_columnconfigure(1, weight=1)
        self.form_card.grid_columnconfigure(2, weight=0)

        CTkLabel(self.form_card, text="Register New Face", font=("Times New Roman", 18, "bold")).grid(
            row=0, column=0, columnspan=4, pady=(10, 5)
        )

        # Hàng 1: Họ tên
        CTkLabel(self.form_card, text="Full Name:", font=("Times New Roman", 14)).grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )
        self.name_entry = CTkEntry(self.form_card, width=250)
        self.name_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w", columnspan=2)

        # Hàng 2: Mã số sinh viên
        CTkLabel(self.form_card, text="Student ID:", font=("Times New Roman", 14)).grid(
            row=2, column=0, padx=10, pady=5, sticky="e"
        )
        self.id_entry = CTkEntry(self.form_card, width=250)
        self.id_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w", columnspan=2)

        # Hàng 3: Auto Capture Switch + Số mẫu + Delay
        self.auto_capture_switch = CTkSwitch(
            self.form_card, text="Auto Capture", command=self.toggle_auto_capture,
            font=("Times New Roman", 14)
        )
        self.auto_capture_switch.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        CTkLabel(self.form_card, text="Samples:", font=("Times New Roman", 12)).grid(
            row=3, column=1, padx=(5,0), pady=5, sticky="e"
        )
        self.samples_option = CTkOptionMenu(
            self.form_card, values=["3", "5", "7", "10"], width=60,
            command=self.change_num_samples
        )
        self.samples_option.set(str(self.NUM_SAMPLES))
        self.samples_option.grid(row=3, column=1, padx=(50,5), pady=5, sticky="w")

        CTkLabel(self.form_card, text="Delay (s):", font=("Times New Roman", 12)).grid(
            row=3, column=2, padx=(5,0), pady=5, sticky="e"
        )
        self.delay_option = CTkOptionMenu(
            self.form_card, values=["1.0", "1.5", "2.0", "2.5"], width=60,
            command=self.change_auto_delay
        )
        self.delay_option.set(str(self.AUTO_DELAY_SEC))
        self.delay_option.grid(row=3, column=2, padx=(50,5), pady=5, sticky="w")

        # Hàng 4: Nút Capture thủ công và Save
        self.capture_btn = CTkButton(
            self.form_card, text="Capture Face", font=("Times New Roman", 14, "bold"),
            image=self.capture_img, compound="left", corner_radius=30,
            command=self.capture_sample, fg_color="#2C7A4D", hover_color="#1E5A38"
        )
        self.capture_btn.grid(row=4, column=0, padx=10, pady=10)

        self.register_btn = CTkButton(
            self.form_card, text="Save Registration", font=("Times New Roman", 14, "bold"),
            image=self.register_img, compound="left", corner_radius=30,
            command=self.register_face, fg_color="#2C7A4D", hover_color="#1E5A38"
        )
        self.register_btn.grid(row=4, column=1, padx=10, pady=10)

        self.sample_count_label = CTkLabel(self.form_card, text=f"Samples: 0/{self.NUM_SAMPLES}", font=("Times New Roman", 12))
        self.sample_count_label.grid(row=4, column=2, padx=10, pady=10)

        # Preview ảnh vừa chụp
        self.preview_label = CTkLabel(
            self.form_card, text="Preview", width=120, height=120,
            fg_color="gray20", corner_radius=10
        )
        self.preview_label.grid(row=5, column=0, columnspan=3, padx=10, pady=5)

        self.progress = CTkProgressBar(self.form_card, mode="indeterminate", height=8)
        self.progress.grid(row=6, column=0, columnspan=3, padx=20, pady=(0, 15), sticky="ew")
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

        self.add_log("Ready. Start webcam, fill info. Enable Auto Capture or use manual Capture.")
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
        if self.webcam_label.winfo_exists():
            self.webcam_label.configure(image="", text="Webcam is Offline")
            self.webcam_label.image = None
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

        # Auto capture logic
        if self.auto_capture_enabled and len(self.embeddings_samples) < self.NUM_SAMPLES:
            self.process_auto_capture(frame)

        # Chuyển đổi để hiển thị
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.webcam_label.configure(image=img_tk)
        self.webcam_label.image = img_tk
        self.after(10, self.update_webcam)

    # -------------------- AUTO CAPTURE LOGIC --------------------
    def process_auto_capture(self, frame):
        """Kiểm tra điều kiện và tự động chụp nếu đủ ổn định"""
        now = datetime.now().timestamp()
        if now - self.last_auto_capture_time < self.AUTO_DELAY_SEC:
            return

        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.DETECTOR_BACKEND,
                enforce_detection=False
            )
        except Exception:
            self.stable_face_counter = 0
            return

        if not faces or len(faces) == 0:
            self.stable_face_counter = 0
            return

        face = faces[0]
        facial_area = face["facial_area"]
        face_w = facial_area["w"]
        face_h = facial_area["h"]
        frame_h, frame_w = frame.shape[:2]
        if face_w < frame_w * 0.2 or face_h < frame_h * 0.2:
            self.stable_face_counter = 0
            return

        # Kiểm tra độ ổn định: khuôn mặt gần trung tâm
        center_x, center_y = frame_w // 2, frame_h // 2
        face_center_x = facial_area["x"] + face_w // 2
        face_center_y = facial_area["y"] + face_h // 2
        offset = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
        max_offset = min(frame_w, frame_h) * 0.15
        if offset > max_offset:
            self.stable_face_counter = 0
            return

        self.stable_face_counter += 1
        if self.stable_face_counter >= 5:
            self.stable_face_counter = 0
            self.last_auto_capture_time = now
            self.capture_sample(frame)

    def toggle_auto_capture(self):
        self.auto_capture_enabled = self.auto_capture_switch.get() == 1
        if self.auto_capture_enabled:
            self.add_log("Auto Capture enabled. Keep your face stable in the center.")
            self.stable_face_counter = 0
            self.last_auto_capture_time = 0
        else:
            self.add_log("Auto Capture disabled.")

    def change_num_samples(self, value):
        self.NUM_SAMPLES = int(value)
        self.sample_count_label.configure(text=f"Samples: {len(self.embeddings_samples)}/{self.NUM_SAMPLES}")
        self.add_log(f"Number of samples changed to {self.NUM_SAMPLES}")

    def change_auto_delay(self, value):
        self.AUTO_DELAY_SEC = float(value)
        self.add_log(f"Auto capture delay set to {self.AUTO_DELAY_SEC} seconds")

    # -------------------- CAPTURE & REGISTRATION --------------------
    def capture_sample(self, frame=None):
        """Chụp một mẫu khuôn mặt, tính embedding và lưu tạm."""
        if not self.is_running or self.cap is None:
            self.add_log("ERROR: Webcam is not running. Please start webcam first.")
            return

        if len(self.embeddings_samples) >= self.NUM_SAMPLES:
            self.add_log(f"Already captured {self.NUM_SAMPLES} samples. Press 'Save Registration'.")
            return

        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                self.add_log("ERROR: Failed to capture frame.")
                return
            frame = cv2.flip(frame, 1)

        self.progress.start()
        self.add_log(f"Capturing sample {len(self.embeddings_samples)+1}/{self.NUM_SAMPLES}...")

        try:
            embedding_objs = DeepFace.represent(
                img_path=frame,
                model_name=self.MODEL_NAME,
                enforce_detection=True,
                detector_backend=self.DETECTOR_BACKEND
            )
            if not embedding_objs:
                self.add_log("No face detected. Please look straight into the camera.")
                self.progress.stop()
                return

            # Lấy embedding và chuẩn hóa L2 (đảm bảo cosine similarity)
            emb = np.array(embedding_objs[0]["embedding"])
            emb = emb / np.linalg.norm(emb)
            facial_area = embedding_objs[0]["facial_area"]

            self.embeddings_samples.append(emb)

            # Cắt vùng mặt để preview
            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                face_pil.thumbnail((120, 120), Image.Resampling.LANCZOS)
                preview_img = CTkImage(light_image=face_pil, dark_image=face_pil, size=(face_pil.width, face_pil.height))
                self.preview_label.configure(image=preview_img, text="")
                self.preview_label.image = preview_img

            self.add_log(f"Sample {len(self.embeddings_samples)} captured successfully.")
            self.sample_count_label.configure(text=f"Samples: {len(self.embeddings_samples)}/{self.NUM_SAMPLES}")

            if len(self.embeddings_samples) == self.NUM_SAMPLES:
                self.add_log("All samples captured. You can now save registration.")
        except Exception as e:
            self.add_log(f"Capture error: {str(e)}")
        finally:
            self.progress.stop()
            self.progress.set(0)

    def register_face(self):
        """Tính embedding trung bình từ các mẫu, chuẩn hóa và lưu vào file."""
        name = self.name_entry.get().strip()
        student_id = self.id_entry.get().strip()
        if not name or not student_id:
            self.add_log("ERROR: Please enter both Full Name and Student ID.")
            return

        if len(self.embeddings_samples) == 0:
            self.add_log("ERROR: No face samples captured. Please capture at least one sample.")
            return

        # Kiểm tra trùng student_id
        existing = self.load_all_embeddings(self.EMBEDDING_DIR)
        if any(reg["student_id"] == student_id for reg in existing):
            self.add_log(f"ERROR: Student ID '{student_id}' already registered. Cannot overwrite.")
            return

        # Tính embedding trung bình và chuẩn hóa
        avg_embedding = np.mean(self.embeddings_samples, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        data = {
            "name": name,
            "student_id": student_id,
            "embedding": avg_embedding,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(self.embeddings_samples)
        }
        filename = f"{student_id}_{name.replace(' ', '_')}.pkl"
        filepath = os.path.join(self.EMBEDDING_DIR, filename)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        self.add_log(f"SUCCESS: Registered {name} (ID: {student_id}) with {len(self.embeddings_samples)} samples.")
        self.add_log(f"Embedding saved to: {filepath}")
        self.reset_registration()

    def reset_registration(self):
        """Xoá dữ liệu mẫu, reset form."""
        self.embeddings_samples = []
        self.sample_count_label.configure(text=f"Samples: 0/{self.NUM_SAMPLES}")
        self.preview_label.configure(image="", text="Preview")
        self.name_entry.delete(0, "end")
        self.id_entry.delete(0, "end")
        self.add_log("Ready for next registration.")

    # -------------------- LOGGING --------------------
    def add_log(self, msg):
        if not self.winfo_exists():
            return
        if hasattr(self, 'log_text') and self.log_text.winfo_exists():
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert("end", f"[{timestamp}] {msg}\n")
            self.log_text.see("end")

    def on_close(self, event=None):
        self.stop_webcam()