# EchoFall-Fall-Severity-Classifier
AI-enabled acoustic monitoring system for detecting fall events and classifying their severity (High, Low, No-Fall). The project uses the SAFE dataset, audio preprocessing, spectrograms, ConvNeXt-Tiny, and Audio Spectrogram Transformer (AST) to support privacy-preserving emergency detection.

Privacy-preserving acoustic fall detection and severity classification using deep learning.

🚀 Project Overview

EchoFall-Fall-Severity-Classifier AI is monitoring system designed to detect fall events and classify their severity into three categories:

High-Risk Fall

Low-Risk Fall

No Fall

The system uses advanced audio processing and deep learning models (ConvNeXt-Tiny & AST) to infer severity from short sound clips while preserving user privacy.

🎧 Features

Spectrogram audio representations

Automatic severity labeling using clustering (K-Means)

Context-aware models using metadata (surface type, body position)

ConvNeXt-Tiny baseline

Audio Spectrogram Transformer (AST) for best performance

Prototype UI for uploading and analyzing audio

📦 Installation
pip install -r requirements.txt

## ▶️ Running the Models

Open the baseline or prototype notebook:

```
baseline_model/convnext-tiny-kmeans.ipynb
prototype_model/ast_+_kmeans.ipynb
```

📊 Prototype Screens

Images in /prototype demonstrate:

Upload audio

Recording state

Analysis in progress

Final severity prediction

📚 Dataset

SAFE Dataset for fall audio events:
https://www.kaggle.com/datasets/antonygarciag/fall-audio-detection-dataset

📝 License

MIT License.

👥 Authors

Joori Shareef

Sarah Aldarwish

Jood Alharbi

Jana Alzobidi

Rahaf Alhodali
