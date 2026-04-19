import customtkinter as ctk
from tkinter import filedialog
import numpy as np
from PIL import Image
import pickle
import os
from datetime import datetime
from deepface import DeepFace
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

class FaceVerificationPage(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs, fg_color="transparent")

        # ===== Đường dẫn model =====
        self.model_path = "app/models/face_mlp.h5"
        self.norm_path = "app/models/norm.pkl"
        self.label_map_path = "app/models/label_map.pkl"
        self.load_models()

        # ===== Biến lưu ảnh & embedding =====
        self.img1_path = None
        self.img2_path = None
        self.embedding1 = None
        self.embedding2 = None
        self.history = []

        # ===== Tạo giao diện =====
        self.setup_ui()

    def load_models(self):
        try:
            self.model = load_model(self.model_path)
            with open(self.norm_path, "rb") as f:
                self.norm = pickle.load(f)
            with open(self.label_map_path, "rb") as f:
                self.label_map = pickle.load(f)
            self.models_loaded = True
        except Exception as e:
            self.models_loaded = False
            print(f"Lỗi tải model: {e}")

    def setup_ui(self):
        # Layout dọc: row0 = Compare (2 ảnh), row1 = Verification Dashboard
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)  # so sánh ảnh
        self.grid_rowconfigure(1, weight=2)  # dashboard

        # ========== ROW 0: COMPARE SECTION ==========
        compare_frame = ctk.CTkFrame(self, corner_radius=15, border_width=2, border_color="gray40")
        compare_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        compare_frame.grid_columnconfigure(0, weight=1)
        compare_frame.grid_columnconfigure(1, weight=1)
        compare_frame.grid_rowconfigure(0, weight=0)
        compare_frame.grid_rowconfigure(1, weight=1)
        compare_frame.grid_rowconfigure(2, weight=0)

        ctk.CTkLabel(compare_frame, text="📷 Image 1", font=("Times New Roman", 18, "bold")).grid(
            row=0, column=0, pady=5)
        ctk.CTkLabel(compare_frame, text="📷 Image 2", font=("Times New Roman", 18, "bold")).grid(
            row=0, column=1, pady=5)

        self.img1_label = ctk.CTkLabel(compare_frame, text="No image", width=250, height=250,
                                       fg_color="gray20", corner_radius=10)
        self.img1_label.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.img2_label = ctk.CTkLabel(compare_frame, text="No image", width=250, height=250,
                                       fg_color="gray20", corner_radius=10)
        self.img2_label.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")

        btn_frame = ctk.CTkFrame(compare_frame, fg_color="transparent")
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        ctk.CTkButton(btn_frame, text="📂 Select Image 1", command=self.load_image_1,
                      font=("Times New Roman", 14), width=140).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="📂 Select Image 2", command=self.load_image_2,
                      font=("Times New Roman", 14), width=140).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="🔍 Compare", command=self.compare_faces,
                      font=("Times New Roman", 14, "bold"), fg_color="#2C7A4D", width=120).pack(side="left", padx=5)

        # ========== ROW 1: VERIFICATION DASHBOARD ==========
        dashboard_frame = ctk.CTkFrame(self, corner_radius=15, border_width=2, border_color="gray40")
        dashboard_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        dashboard_frame.grid_columnconfigure(0, weight=1)
        dashboard_frame.grid_rowconfigure(0, weight=0)   # tiêu đề
        dashboard_frame.grid_rowconfigure(1, weight=1)   # biểu đồ
        dashboard_frame.grid_rowconfigure(2, weight=0)   # metrics
        dashboard_frame.grid_rowconfigure(3, weight=1)   # lịch sử

        ctk.CTkLabel(dashboard_frame, text="📊 Verification Dashboard", font=("Times New Roman", 20, "bold")).grid(
            row=0, column=0, pady=10)

        # --- Biểu đồ thanh confidence ---
        self.fig = plt.Figure(figsize=(6, 3), dpi=80, facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.set_title("Confidence Score", color='white', fontsize=12)
        self.canvas = FigureCanvasTkAgg(self.fig, master=dashboard_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # --- Metrics frame ---
        metrics_frame = ctk.CTkFrame(dashboard_frame, corner_radius=10)
        metrics_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)

        self.similarity_label = ctk.CTkLabel(metrics_frame, text="Cosine Similarity: --",
                                             font=("Times New Roman", 14))
        self.similarity_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.euclidean_label = ctk.CTkLabel(metrics_frame, text="Euclidean Distance: --",
                                            font=("Times New Roman", 14))
        self.euclidean_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.prediction_label = ctk.CTkLabel(metrics_frame, text="Prediction: --",
                                             font=("Times New Roman", 14, "bold"))
        self.prediction_label.grid(row=1, column=0, columnspan=2, pady=5)

        # --- Lịch sử so sánh ---
        history_frame = ctk.CTkFrame(dashboard_frame, corner_radius=10)
        history_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")
        ctk.CTkLabel(history_frame, text="📜 Recent Comparisons", font=("Times New Roman", 16, "bold")).pack(pady=5)
        self.history_text = ctk.CTkTextbox(history_frame, font=("Courier New", 11), height=120)
        self.history_text.pack(fill="both", expand=True, padx=5, pady=5)

    # -------------------- XỬ LÝ ẢNH --------------------
    def load_image_1(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img1_path = path
            self.show_image(path, self.img1_label)
            self.embedding1 = None

    def load_image_2(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img2_path = path
            self.show_image(path, self.img2_label)
            self.embedding2 = None

    def show_image(self, path, label_widget):
        img = Image.open(path)
        img.thumbnail((250, 250), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
        label_widget.configure(image=ctk_img, text="")
        label_widget.image = ctk_img

    def get_embedding(self, img_path):
        rep = DeepFace.represent(img_path=img_path, model_name="Facenet512",
                                 enforce_detection=False, detector_backend="opencv")
        if not rep:
            raise ValueError("No face detected")
        emb = np.array(rep[0]["embedding"]).reshape(1, -1)
        emb = self.norm.transform(emb)
        return emb.flatten()

    def compute_similarity(self, emb1, emb2):
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        euclidean = np.linalg.norm(emb1 - emb2)
        return cos_sim, euclidean

    def compare_faces(self):
        if not self.models_loaded:
            self.prediction_label.configure(text="Model not loaded!", text_color="red")
            return
        if not self.img1_path or not self.img2_path:
            self.prediction_label.configure(text="Please select both images.", text_color="orange")
            return

        def process():
            try:
                if self.embedding1 is None:
                    self.embedding1 = self.get_embedding(self.img1_path)
                if self.embedding2 is None:
                    self.embedding2 = self.get_embedding(self.img2_path)

                emb1_2d = self.embedding1.reshape(1, -1)
                emb2_2d = self.embedding2.reshape(1, -1)

                pred1 = self.model.predict(emb1_2d, verbose=0)
                pred2 = self.model.predict(emb2_2d, verbose=0)
                label1 = np.argmax(pred1)
                label2 = np.argmax(pred2)
                conf1 = np.max(pred1)
                conf2 = np.max(pred2)

                name1 = self.label_map.get(label1, "Unknown")
                name2 = self.label_map.get(label2, "Unknown")

                cos_sim, euclidean = self.compute_similarity(self.embedding1, self.embedding2)

                self.after(0, lambda: self.update_dashboard(name1, name2, conf1, conf2, cos_sim, euclidean, label1 == label2))

                timestamp = datetime.now().strftime("%H:%M:%S")
                result = "✅ SAME" if label1 == label2 else "❌ DIFFERENT"
                history_entry = f"[{timestamp}] {name1} vs {name2} → {result} (sim={cos_sim:.2f})"
                self.history.insert(0, history_entry)
                if len(self.history) > 10:
                    self.history.pop()
                self.after(0, self.update_history)

            except Exception as e:
                self.after(0, lambda: self.prediction_label.configure(text=f"Error: {str(e)}", text_color="red"))

        threading.Thread(target=process, daemon=True).start()

    def update_dashboard(self, name1, name2, conf1, conf2, cos_sim, euclidean, is_same):
        self.ax.clear()
        categories = [f"Image 1\n{name1}", f"Image 2\n{name2}"]
        confidences = [conf1, conf2]
        colors = ['#2C7A4D' if is_same else '#D32F2F', '#2C7A4D' if is_same else '#D32F2F']
        bars = self.ax.bar(categories, confidences, color=colors, alpha=0.8)
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Confidence", color='white')
        self.ax.set_title("Prediction Confidence", color='white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        for bar, conf in zip(bars, confidences):
            self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f"{conf:.2f}", ha='center', color='white', fontsize=10)
        self.canvas.draw()

        self.similarity_label.configure(text=f"Cosine Similarity: {cos_sim:.4f}")
        self.euclidean_label.configure(text=f"Euclidean Distance: {euclidean:.4f}")
        if is_same:
            self.prediction_label.configure(text=f"✅ SAME PERSON: {name1}", text_color="green")
        else:
            self.prediction_label.configure(text=f"❌ DIFFERENT: {name1} vs {name2}", text_color="red")

    def update_history(self):
        self.history_text.delete("1.0", "end")
        for entry in self.history[:8]:
            self.history_text.insert("end", entry + "\n")