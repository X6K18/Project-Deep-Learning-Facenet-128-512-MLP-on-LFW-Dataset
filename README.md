#  Face Recognition using FaceNet (128/512) + MLP on LFW Dataset

##  Introduction
This project implements a **face recognition system** using **FaceNet embeddings (128D / 512D)** combined with a **Multi-Layer Perceptron (MLP)** classifier.

The model is trained and evaluated on the **Labeled Faces in the Wild (LFW)** dataset, a popular benchmark in face recognition with real-world variations such as lighting, pose, and expressions.

---

##  Dataset
- **Name:** LFW (Labeled Faces in the Wild)  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/atulanandjha/lfwpeople  

### Dataset Details:
- ~13,000+ images  
- ~5,700 identities  
- Images collected in unconstrained environments  

---

##  Methodology

### 1. Face Embedding (Feature Extraction)
We use **FaceNet** to convert face images into vector representations:
- **FaceNet128** → 128-dimensional embedding  
- **FaceNet512** → 512-dimensional embedding  

---

### 2. Classification (MLP)
A **Multi-Layer Perceptron (MLP)** is used for classification:
- Input: Face embeddings  
- Hidden layers: Dense + ReLU  
- Output: Softmax (identity prediction)  

---
