# 📊 Results & Final System Output

## 🎯 Overview
The system has been fully developed with a user-friendly interface (UI), supporting:
- Real-time face recognition
- Face verification
- Automatic attendance tracking
- Time-based status control

---

## 🖥️ System Interface

UI Demo

> 📌 *Note:* Replace `results/nguyen_tung.png` with your actual image path in the repository.

---

## 🧩 Main Features

### 1. Face Verification
- Verify identity for a specific individual
- Display similarity/confidence score (%)

### 2. Face Recognition
- Recognize multiple faces simultaneously
- Display:
  - Name
  - Confidence score

### 3. Register Face
- Register new faces into the system
- Store embeddings for future recognition

### 4. Attendance System
- Automatic attendance via webcam
- Prevent duplicate marking ("Already marked")
- Store real-time attendance logs

---

## ⚙️ System Status

The system displays:
- 🕒 Current time  
- ⏱️ Valid time range  
- 🚫 System status:
  - `IN TIME`
  - `OUT OF TIME`

📌 Example:


---

## 👤 Recognition Results

From the demo image:

- Person 1:
  - Name: `tung`
  - Confidence: `94.5%`
  - Status: Already marked

- Person 2:
  - Name: `nguyen`
  - Confidence: `87.1%`
  - Status: Already marked

👉 The system performs reliably with multiple faces in a single frame.

---

## 📋 Attendance Table (Today's Attendance)

| Name   | Time     | Status |
|--------|----------|--------|
| nguyen | 16:26:15 | Off    |
| tung   | 16:26:24 | Off    |

📌 Automatically updated in real time.

---

## 🚀 Achievements

- ✅ Real-time recognition with high accuracy (~85–95%)
- ✅ Multi-face detection and recognition
- ✅ User-friendly UI (CustomTkinter)
- ✅ Automated attendance system
- ✅ Time-based access control

---

## ⚠️ Limitations

- Low lighting conditions may reduce accuracy  
- Large face angles (pose variation) affect performance  
- Not fully optimized for low-performance devices  

---

## 🔮 Future Work

- Integrate IP camera / IoT systems  
- Deploy as a web application (Flask / FastAPI)  
- Improve accuracy using ArcFace  
- Store data in databases (MySQL, MongoDB)  

---

## 🎉 Conclusion

The system successfully implements the full pipeline:

> **Face Detection → FaceNet → MLP → Recognition → Attendance**

This solution can be applied in real-world scenarios such as:
- Student attendance systems  
- Access control systems  
- Security and surveillance  

---