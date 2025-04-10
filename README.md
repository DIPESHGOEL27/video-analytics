# 🎥 AI-Powered Video Analytics System 🚀

An industry-ready, modular system for real-time **object detection**, **multi-object tracking**, and **threat analytics** powered by **YOLOv8** and **Deep SORT** — with a beautiful and interactive **Streamlit Dashboard**.

---

## 🔍 Features

- ✅ **YOLOv8-based Object Detection**
- 🧠 **Deep SORT Tracker with Appearance Features**
- 🔁 **Real-time Frame-by-Frame Processing**
- 📊 **Streamlit Dashboard with Upload, Preview, and Download**
- 🚨 **Custom Alert Zones** to trigger visual warnings
- 🛠️ **Modular, Scalable Codebase** suitable for production


---

## 🛠️ Setup

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/DIPESHGOEL27/video-analytics.git
cd video-analytics
```

### 2️⃣ Create & Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ Ensure `torch` is installed according to your GPU/CPU setup:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4️⃣ Download Pretrained Models
- [YOLOv8 weights](https://github.com/ultralytics/ultralytics)
- `mars-small128.pb` appearance model in:
  ```
  app/tracking/deep_sort/resources/networks/mars-small128.pb
  ```

---

## 🚀 Run the App

```bash
streamlit run main.py
```

Upload any video, track objects live, and download processed output with overlays.

---

## ⚙️ Configuration (config.yaml)

```yaml
video_input: "data/input_video.mp4"
video_output: "output/output_video.mp4"

yolo:
  model_path: "app/models/yolov8n.pt"
  confidence_threshold: 0.5

deep_sort:
  model_path: "app/tracking/deep_sort/resources/networks/mars-small128.pb"
  max_age: 30
  n_init: 3
  max_cosine_distance: 0.4
  nn_budget: 100

alert:
  enabled: true
  alert_zone: [100, 100, 400, 400]
```

---

## 📦 Deployment

To deploy this system:
- Host on **Streamlit Cloud**, **Render**, or **Docker**.
- Upload only `.py` files and model weights (not venv or cache folders).
- Ensure `.gitignore` includes `__pycache__/`, `*.pyc`, `venv/`.


---

## 👨‍💻 Developed by

**Dipesh Goel**  
`Third-year undergrad @ IIT Kharagpur`  
💼 Ocean Engineering and Naval Architecture  
🌐 [LinkedIn](https://www.linkedin.com/in/dipesh-goel/) • ✉️ dipeshgoel27@gmail.com

---

## 📃 License

This project is licensed under the **MIT License**.

---
