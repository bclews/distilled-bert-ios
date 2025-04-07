# 🧠 Running a BERT Model on an iPhone

This project explores the process of adapting a large, pre-trained BERT model—originally designed for GPU-rich environments—for efficient **on-device inference** on an iPhone. It covers the full journey: model conversion, distillation, Core ML integration, and iOS app development using SwiftUI.

The work was completed during a 3-day internal hackathon focused on personal growth and technical exploration. This repository **does not contain** any models or data. Consequently, loading models and trying to run the various scripts and apps will result in errors. This repository is intended to serve as a reference for those interested in the technical aspects of the project.

---

## 📚 Related Blog Post

For a detailed technical breakdown and learnings from the project, check out the blog post:
[Running a BERT Model on an iPhone](https://clews.id.au/posts/running-a-bert-model-on-an-iphone-a-three-day-journey-from-data-center-to-pocket/)

---

## 📁 Project Structure

```
.
├── mobile_deployment/                 # iOS app code and Core ML model
├── model_development/                 # Model training and conversion code
  ├── bert_to_onnx_to_coreML.py      # Conversion pipeline from PyTorch → ONNX → Core ML
  ├── distillation_data_prep.py      # Preprocessing and data augmentation for distillation
  ├── knowledge_distillation.py      # Student-teacher training pipeline
  ├── models/                        # Folder for model checkpoints and exports
  ├── data/                          # Training data and distillation splits
  └── requirements.txt               # Python dependencies
```

---

## 🚀 Getting Started

### 🔧 Environment Setup (Python)

1. Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r model_development/requirements.txt
   ```

### 🏗 Model Development

- Convert BERT model to Core ML:

  ```bash
  python model_development/bert_to_onnx_to_coreML.py
  ```

- Run knowledge distillation:

  ```bash
  python model_development/knowledge_distillation.py
  ```

- Prepare training data:

  ```bash
  python model_development/distillation_data_prep.py
  ```

### 📱 Mobile Deployment

1. Open the Xcode project:

   ```
   open mobile_deployment/auto_tagger/auto_tagger.xcodeproj
   ```

2. Build and run on an iPhone simulator or real device.

   Make sure `distilled_model.mlpackage` and `vocab.txt` are included in your Xcode project.

---

## 📊 Performance

| Model Version      | Size   | Accuracy (Approx) | Mobile Inference |
|--------------------|--------|-------------------|------------------|
| BERT-large (Full)  | ~670MB | ✅ High            | ⚠️ Slower        |
| Distilled BERT     | ~134MB | 🟡 Moderate        | ✅ Faster        |
