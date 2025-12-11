EchoFall – Fall Severity Classifier

A privacy-preserving acoustic monitoring system for detecting fall events and classifying their severity into High, Low, or No-Fall.
The project uses the SAFE dataset, audio preprocessing pipelines, spectrogram representations, and two deep-learning models: ConvNeXt-Tiny (baseline) and Audio Spectrogram Transformer (AST) (prototype).

🚀 Overview

EchoFall is designed to analyze short audio clips and determine whether a fall has occurred, followed by estimating its severity.
The goal is to provide an AI-powered safety tool that works without cameras and protects user privacy.

The project consists of three main components:

Baseline Model: ConvNeXt-Tiny 

Prototype Model: AST 

Interactive UI Prototype: Upload, record, and analyze fall-sound samples

🎧 Features

Audio preprocessing and spectrogram generation

Automatic severity labeling through K-Means clustering

Context-aware severity inference using metadata

ConvNeXt-Tiny baseline model

AST model for improved performance

Lightweight prototype UI demo

📂 Repository Structure
baseline_model/
   └── convnext_tiny_kmeans.ipynb      # Baseline model notebook

prototype_model/
   └── ast_kmeans_pipeline.ipynb       # Prototype AST model

experiments/
   └── notebooks/                      # Additional model trials (future)

prototype_ui/
   └── ui_demo_images/                 # Screenshots for the demo UI

▶️ Running the Models

Open the notebooks directly through Colab or Jupyter or Kaggle:

baseline_model/convnext_tiny_kmeans.ipynb
prototype_model/ast_kmeans_pipeline.ipynb


Each notebook includes:

Preprocessing steps

Spectrogram generation

Model training

Evaluation metrics

🖥️ Prototype Demo

UI demo link:
https://echofall-ai-demo.lovable.app/

The prototype interface supports:

Uploading audio samples

Recording from the browser

Running fall-severity analysis

Displaying the final prediction

Screenshots are included under:
prototype_ui/

📚 Dataset

SAFE: Fall audio dataset
https://www.kaggle.com/datasets/antonygarciag/fall-audio-detection-dataset

📝 License

MIT License

👥 Authors

Joori Shareef

Sarah Aldarwish

Jood Alharbi

Jana Alzobidi

Rahaf Alhodali
