"""
The notebook implements a multimodal fall-severity estimation pipeline.
First, it extracts spectral audio features and encodes contextual descriptors such as position and surface.
A Kmeans Model then clusters fall events into two natural groups representing High and Low severity, using both context and audio jointly.
These data-driven severity labels are used to train a deep neural network consisting of a ConvNeXt-Tiny branch for spectrogram analysis and a meta-MLP for contextual information.
The fused model learns how context modifies the meaning of acoustic impact patterns and predicts severity from raw audio and metadata.
"""

# Block 0 — setup

!pip install --quiet torch torchvision torchaudio librosa scikit-learn pandas matplotlib

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import random
from sklearn.model_selection import train_test_split
import pathlib
from PIL import Image
from sklearn.metrics import  ConfusionMatrixDisplay
import matplotlib.pyplot as plt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

"""# load .wav files and parse filename"""

# Block 1 — load all .wav files and parse metadata from filename

DATA_DIR = "/kaggle/input/fall-audio-detection-dataset"

records = []

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.lower().endswith(".wav"):
            path = os.path.join(root, f)
            base = os.path.splitext(f)[0]           # 'AA-BBB-CC-DDD-FF'
            try:
                aa, bbb, cc, ddd, ff = base.split("-")
            except ValueError:
                print("Filename not matching pattern:", f)
                continue

            binary_label = "Fall" if ff == "01" else "No-Fall"

            records.append({
                "filename": f,
                "filepath": path,
                "AA": aa,
                "CC": cc,
                "FF": ff,
                "Binary_Label": binary_label,
            })

df = pd.DataFrame(records)
print("Total audio files:", len(df))
df.head()
#result = dataframe df with 950 rows.

"""# map CC to context (location / surface / position)"""

# Block 2 — environment mapping (CC -> location, surface, position)

env_map = {
    "01": {"location": "Basement", "surface": "Carpet over concrete", "position": "Lying"},
    "02": {"location": "Basement", "surface": "Carpet over concrete", "position": "Standing"},
    "03": {"location": "Lab",      "surface": "Carpet over wood",     "position": "Lying"},
    "04": {"location": "Lab",      "surface": "Carpet over wood",     "position": "Standing"},
    "05": {"location": "Stairs",   "surface": "Concrete",             "position": "Standing"},
    "06": {"location": "Wood",     "surface": "Wood",                 "position": "Lying"},
    "07": {"location": "Wood",     "surface": "Wood",                 "position": "Standing"},
    # if dataset has more CCs, add them here
}

def get_env(cc, key):
    info = env_map.get(cc)
    if info is None:
        return "Unknown"
    return info.get(key, "Unknown")

df["location"] = df["CC"].map(lambda x: get_env(x, "location"))
df["surface"]  = df["CC"].map(lambda x: get_env(x, "surface"))
df["position"] = df["CC"].map(lambda x: get_env(x, "position"))

# numerical encodings for the CNN meta branch
df["position_enc"] = df["position"].map({"Standing": 0, "Lying": 1}).fillna(-1).astype(int)
df["surface_enc"]  = df["surface"].astype("category").cat.codes

df.head()
#So now each file has:
#Context (semantic): location/surface/position.
#Context (numeric): position_enc (0 / 1) and surface_enc (categorical code).

"""# hand-crafted audio features"""

from tqdm import tqdm
def compute_energy(y):
    return float(np.sum(y ** 2))

def compute_centroid(y, sr):
    return float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())

def compute_bandwidth(y, sr):
    return float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())

def compute_rolloff(y, sr):
    return float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())

def compute_flatness(y):
    return float(librosa.feature.spectral_flatness(y=y).mean())

def compute_flux(y):
    S = np.abs(librosa.stft(y))
    flux = librosa.onset.onset_strength(S=S)
    return float(np.mean(flux))

def compute_hf_ratio(y, sr):
    # Ratio of energy above 2 kHz
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    hf_mask = freqs > 2000
    hf_energy = S[hf_mask].sum()
    total_energy = S.sum() + 1e-8
    return float(hf_energy / total_energy)


# Compute features for ALL samples
feature_rows = []
TARGET_SR = 16000   # temporary; AST will override later but OK for feature extraction

for i, row in tqdm(df.iterrows(), total=len(df)):
    y, sr = librosa.load(row["filepath"], sr=TARGET_SR)

    energy      = compute_energy(y)
    centroid    = compute_centroid(y, sr)
    bandwidth   = compute_bandwidth(y, sr)
    rolloff     = compute_rolloff(y, sr)
    flatness    = compute_flatness(y)
    flux        = compute_flux(y)
    hf_ratio    = compute_hf_ratio(y, sr)

    feature_rows.append({
        "filepath": row["filepath"],
        "energy": energy,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness": flatness,
        "flux": flux,
        "hf_ratio": hf_ratio,

        # context encodings from Block 2
        "position_enc": row["position_enc"],
        "surface_enc": row["surface_enc"],

        # binary fall label from Block 1
        "Binary_Label": row["Binary_Label"],
    })

feature_df = pd.DataFrame(feature_rows)
feature_df.head()

full_df = feature_df.merge(df[["filepath", "label"]], on="filepath", how="left")

print("Columns in full_df:", full_df.columns)

feature_cols = [
    "energy",
    "hf_ratio",
    "centroid",
    "bandwidth",
    "rolloff",
    "flatness",
    "flux",
    "position_enc",
    "surface_enc",
]

label_col = "label"

print("=== ANOVA results for High vs Low vs No-Fall ===\n")

for feat in feature_cols:
    if feat not in full_df.columns:
        print(f"Feature {feat} not found in full_df → skipping.\n")
        continue

    high_vals = full_df[full_df[label_col] == "High"][feat].values
    low_vals  = full_df[full_df[label_col] == "Low"][feat].values
    no_vals   = full_df[full_df[label_col] == "No-Fall"][feat].values

    if len(high_vals) == 0 or len(low_vals) == 0 or len(no_vals) == 0:
        print(f"Feature {feat}: one of the groups is empty → skipping.\n")
        continue

    F, p = f_oneway(high_vals, low_vals, no_vals)

    print(f"Feature: {feat}")
    print(f"  F-statistic = {F:.4f}")
    print(f"  p-value     = {p:.8f}")

    if p < 0.05:
        print("  → Significant difference ✔ (good)\n")
    else:
        print("  → NOT significant ✖ (classes overlap)\n")

"""# context + audio KMEANS  clustering"""

falls = feature_df[feature_df["Binary_Label"] == "Fall"].copy()

feature_cols = [
    "position_enc",
    "surface_enc",
    "energy",
    "hf_ratio",
    "centroid",
    "bandwidth",
    "rolloff",
    "flatness",
    "flux"
]

X_falls = falls[feature_cols].values

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

falls["kmeans_cluster"] = kmeans.fit_predict(X_falls)

cluster_means = falls.groupby("kmeans_cluster")["energy"].mean()
high_cluster = cluster_means.idxmax()

falls["risk_label"] = falls["kmeans_cluster"].apply(
    lambda c: "High" if c == high_cluster else "Low"
)

sil_score = silhouette_score(X_falls, falls["kmeans_cluster"])
print("Silhouette score (KMeans, falls only):", sil_score)

plt.figure(figsize=(7,5))

for label, color in zip(["High", "Low"], ["tab:red", "tab:blue"]):
    subset = falls[falls["risk_label"] == label]

    plt.scatter(
        subset["energy"],
        subset["hf_ratio"],
        alpha=0.6,
        s=30,
        c=color,
        label=label
    )

plt.xlabel("Energy")
plt.ylabel("High-frequency ratio")
plt.title("K-Means Risk Clusters")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

"""# FINAL LABEL MERGING (High / Low / No-Fall)"""

# Map risk labels only for fall samples
risk_map = dict(zip(falls["filepath"], falls["risk_label"]))

final_labels = []
for i, row in df.iterrows():
    if row["Binary_Label"] == "No-Fall":
        final_labels.append("No-Fall")
    else:
        final_labels.append(risk_map[row["filepath"]])

df["label"] = final_labels
df.head()

falls = feature_df[feature_df["Binary_Label"] == "Fall"].copy()

feature_cols = [
    "position_enc",
    "surface_enc",
    "energy",
    "hf_ratio",
    "centroid",
    "bandwidth",
    "rolloff",
    "flatness",
    "flux",
]

X_falls = falls[feature_cols].values

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

falls["kmeans_cluster"] = kmeans.fit_predict(X_falls)

cluster_means = falls.groupby("kmeans_cluster")["energy"].mean()
high_cluster = cluster_means.idxmax()

falls["risk_label"] = falls["kmeans_cluster"].apply(
    lambda c: "High" if c == high_cluster else "Low"
)

print(cluster_means)
print("High cluster index:", high_cluster)

sil_kmeans = silhouette_score(X_falls, falls["kmeans_cluster"])
print("Silhouette score (KMeans, falls only):", sil_kmeans)

risk_map = dict(zip(falls["filepath"], falls["risk_label"]))

final_labels = []
for _, row in df.iterrows():
    if row["Binary_Label"] == "No-Fall":
        final_labels.append("No-Fall")
    else:
        final_labels.append(risk_map.get(row["filepath"], "Low"))

df["label"] = final_labels

df["label"].value_counts()

labels_to_show = ["High", "Low", "No-Fall"]

plt.figure(figsize=(15, 4))

for i, lab in enumerate(labels_to_show, start=1):
    subset = df[df["label"] == lab]

    if subset.empty:
        print(f"No samples found for label: {lab}")
        continue

    sample_row = subset.sample(1, random_state=0)
    file_path = sample_row["filepath"].iloc[0]

    print(f"{lab} example file:", sample_row["filename"].iloc[0])

    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR)
    except NameError:
        y, sr = librosa.load(file_path, sr=16000)

    # STFT
    n_fft = 1024
    hop_length = 256
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)


    plt.subplot(1, 3, i)
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
    )
    plt.title(f"{lab} sample", fontsize=12)
    plt.colorbar(format="%+2.0f dB")

plt.tight_layout()
plt.show()

"""# AA fold train/val split"""

RANDOM_SEED = 42

def stratified_train_test(df, test_size=0.20, label_col="label", seed=RANDOM_SEED):
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=seed
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

df_train, df_val = stratified_train_test(df, label_col="label")

print("Train set counts:")
print(df_train["label"].value_counts(), "\n")

print("Test set counts:")
print(df_val["label"].value_counts(), "\n")

"""# STFT spectrogram generation"""

# Block 6 — generate STFT spectrogram images for each split + label

SPEC_ROOT = "dataset_spectrograms"

def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def save_spectrogram(row, split_name):
    y, sr = librosa.load(row["filepath"], sr=22050)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    label = row["label"]
    base = os.path.splitext(os.path.basename(row["filepath"]))[0] + ".png"
    out_dir = os.path.join(SPEC_ROOT, split_name, label)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, base)

    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis("off")
    librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return out_path

# generate for train
df_train["spec_path"] = [save_spectrogram(r, "train") for _, r in df_train.iterrows()]
df_val["spec_path"]   = [save_spectrogram(r, "val") for _, r in df_val.iterrows()]

"""# ConvNeXtWithMeta model

CNN learns patterns in time–frequency energy (STFT image).
MLP learns how position + surface correlate with risk.
Final linear layer learns how to weight both feature sets to predict High, Low, No-Fall.
There is no “first context then audio”; it’s a joint feature vector and the final linear layer learns weights for all dimensions.
"""

# Block 7 — ConvNeXtWithMeta definition
#pretrained ConvNeXt-Tiny backbone
class ConvNeXtWithMeta(nn.Module):
    def __init__(self, num_classes, meta_dim=2):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        backbone = convnext_tiny(weights=weights)

        # adapt first conv for 1-channel input
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        backbone.features[0][0] = new_conv

        in_feats = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Identity()

        self.backbone = backbone
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(in_feats + 16, num_classes)

#Forward
    def forward(self, x, meta):
        feats = self.backbone(x)          # (B, in_feats)
        meta_feats = self.meta_mlp(meta)  # (B, 16)
        fused = torch.cat([feats, meta_feats], dim=1)
        logits = self.classifier(fused)
        return logits

"""# Dataset, transforms, DataLoaders"""

# Block 8 — dataset class that returns (image, meta, label_idx)

CLASSES = ["High", "Low", "No-Fall"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

class SpecMetaDataset(Dataset):
    def __init__(self, df_split, transform=None):
        self.df = df_split.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["spec_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        meta = torch.tensor(
            [row["position_enc"], row["surface_enc"]],
            dtype=torch.float32
        )
        label_idx = class_to_idx[row["label"]]

        # Return only simple types; keep position/surface as strings if you need them later
        return img, meta, label_idx, row["position"], row["surface"]


train_ds = SpecMetaDataset(df_train, transform=train_tf)
val_ds   = SpecMetaDataset(df_val, transform=val_tf)

BATCH_SIZE = 8
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

len(train_ds), len(val_ds)

"""# training, validation, evaluation"""

# Block 9 — training loop with history
"""Loss = CrossEntropy over 3 classes.
Optimizer = AdamW.
Save best model by validation accuracy.
Then reload and compute classification_report and confusion matrix."""

EPOCHS = 15
LR = 1e-4
WD = 1e-4

model = ConvNeXtWithMeta(num_classes=len(CLASSES), meta_dim=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

best_val_acc = 0.0
train_hist = []
val_hist = []

for epoch in range(1, EPOCHS+1):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for xb, mb, yb, _, _ in train_dl:
        xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb, mb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * xb.size(0)
        correct  += (logits.argmax(1) == yb).sum().item()
        total    += yb.size(0)

    train_acc = correct / total
    train_hist.append(train_acc)

    # validation
    model.eval()
    v_total, v_correct = 0, 0
    with torch.no_grad():
        for xb, mb, yb, _, _ in val_dl:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            out = model(xb, mb)
            v_correct += (out.argmax(1) == yb).sum().item()
            v_total   += yb.size(0)
    val_acc = v_correct / v_total
    val_hist.append(val_acc)

    print(f"Epoch {epoch:02d}: train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_convnext_meta.pth")

print("Best val acc:", best_val_acc)

# Plot training vs validation accuracy

plt.figure(figsize=(6,4))
plt.plot(train_hist, label="Train Acc")
plt.plot(val_hist, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training / Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

model.load_state_dict(torch.load("best_convnext_meta.pth", map_location=DEVICE))
model.eval()

all_true = []
all_pred = []

with torch.no_grad():
    for xb, mb, yb, _, _ in val_dl:     # <-- must match your dataset return format
        xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
        out = model(xb, mb)
        preds = out.argmax(1)

        all_true.extend(yb.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())

print(classification_report(all_true, all_pred, target_names=CLASSES))

cm = confusion_matrix(all_true, all_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Validation Set)")
plt.show()