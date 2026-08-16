# ==========================================
# Fashion Product Category Prediction
# Model Evaluation
# ==========================================

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from transformers import AutoTokenizer, AutoModel


# ==========================================
# 1. Configuration
# ==========================================

DATA_PATH = "dataset/styles.csv"
IMAGE_DIR = "dataset/images"

MODEL_DIR = "saved_model"

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR,
    "multimodal_classifier.keras"
)

LABEL_MAPPING_PATH = os.path.join(
    MODEL_DIR,
    "label_mapping.json"
)

IMAGE_SIZE = (224, 224)

IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE = 16

MAX_LENGTH = 32

TEXT_MODEL_NAME = "distilbert-base-uncased"

RANDOM_STATE = 42


# ==========================================
# 2. Load Dataset
# ==========================================

print("\n==========================================")
print("LOADING DATASET")
print("==========================================")

df = pd.read_csv(
    DATA_PATH,
    engine="python",
    on_bad_lines="skip"
)

df = df[
    [
        "id",
        "productDisplayName",
        "articleType"
    ]
]

df = df.dropna().copy()


# ==========================================
# 3. Select Same Top 10 Categories
# ==========================================

top_categories = (
    df["articleType"]
    .value_counts()
    .head(10)
    .index
)

df = df[
    df["articleType"].isin(top_categories)
].copy()


# ==========================================
# 4. Create Image Paths
# ==========================================

df["image_path"] = (
    df["id"].astype(str) + ".jpg"
)

df["image_path"] = df["image_path"].apply(
    lambda x: os.path.join(
        IMAGE_DIR,
        x
    )
)


# ==========================================
# 5. Keep Existing Images
# ==========================================

df = df[
    df["image_path"].apply(os.path.exists)
].copy()


# ==========================================
# 6. Encode Labels
# ==========================================

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(
    df["articleType"]
)


# ==========================================
# 7. SAME Train / Validation / Test Split
# ==========================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["label"]
)


print("\nTest Samples:")
print(len(test_df))


# ==========================================
# 8. Load Saved ANN
# ==========================================

print("\n==========================================")
print("LOADING SAVED CLASSIFIER")
print("==========================================")

classifier = tf.keras.models.load_model(
    CLASSIFIER_PATH
)

print("Classifier loaded successfully.")


# ==========================================
# 9. Load MobileNetV2
# ==========================================

print("\nLoading MobileNetV2...")

image_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

image_model.trainable = False

print("MobileNetV2 loaded.")


# ==========================================
# 10. Load DistilBERT
# ==========================================

print("\nLoading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    TEXT_MODEL_NAME
)

text_model = AutoModel.from_pretrained(
    TEXT_MODEL_NAME
)

text_model.eval()


# ==========================================
# 11. Select Device
# ==========================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


text_model = text_model.to(device)

print("DistilBERT device:", device)


# ==========================================
# 12. Image Feature Extraction
# ==========================================

def extract_image_features(dataframe):

    features = []

    image_paths = dataframe[
        "image_path"
    ].tolist()

    total = len(image_paths)

    print(
        "\nExtracting test image features..."
    )

    for start in range(
        0,
        total,
        IMAGE_BATCH_SIZE
    ):

        batch_paths = image_paths[
            start:start + IMAGE_BATCH_SIZE
        ]

        batch_images = []

        for path in batch_paths:

            image = Image.open(
                path
            ).convert("RGB")

            image = image.resize(
                IMAGE_SIZE
            )

            image = np.array(
                image,
                dtype=np.float32
            )

            image = preprocess_input(
                image
            )

            batch_images.append(
                image
            )

        batch_images = np.array(
            batch_images,
            dtype=np.float32
        )

        batch_features = image_model.predict(
            batch_images,
            verbose=0
        )

        features.append(
            batch_features
        )

        processed = min(
            start + IMAGE_BATCH_SIZE,
            total
        )

        print(
            f"\rProcessed {processed}/{total}",
            end=""
        )

    print()

    return np.vstack(features)


# ==========================================
# 13. Text Feature Extraction
# ==========================================

def extract_text_features(dataframe):

    features = []

    texts = (
        dataframe[
            "productDisplayName"
        ]
        .astype(str)
        .tolist()
    )

    total = len(texts)

    print(
        "\nExtracting test text features..."
    )

    for start in range(
        0,
        total,
        TEXT_BATCH_SIZE
    ):

        batch_texts = texts[
            start:start + TEXT_BATCH_SIZE
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = text_model(
                **encoded
            )

        hidden_states = (
            outputs.last_hidden_state
        )

        attention_mask = (
            encoded["attention_mask"]
            .unsqueeze(-1)
            .float()
        )

        masked_hidden_states = (
            hidden_states * attention_mask
        )

        summed = masked_hidden_states.sum(
            dim=1
        )

        counts = attention_mask.sum(
            dim=1
        )

        text_embeddings = (
            summed /
            counts.clamp(min=1e-9)
        )

        text_embeddings = (
            text_embeddings
            .cpu()
            .numpy()
        )

        features.append(
            text_embeddings
        )

        processed = min(
            start + TEXT_BATCH_SIZE,
            total
        )

        print(
            f"\rProcessed {processed}/{total}",
            end=""
        )

    print()

    return np.vstack(features)


# ==========================================
# 14. Extract Test Features
# ==========================================

X_test_image = extract_image_features(
    test_df
)

X_test_text = extract_text_features(
    test_df
)


# ==========================================
# 15. Feature Fusion
# ==========================================

X_test = np.concatenate(
    [
        X_test_image,
        X_test_text
    ],
    axis=1
)

y_test = test_df["label"].values


print("\n==========================================")
print("FEATURE INFORMATION")
print("==========================================")

print(
    "Image Features:",
    X_test_image.shape
)

print(
    "Text Features:",
    X_test_text.shape
)

print(
    "Fused Features:",
    X_test.shape
)


# ==========================================
# 16. Make Predictions
# ==========================================

print("\n==========================================")
print("MAKING PREDICTIONS")
print("==========================================")

probabilities = classifier.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)


# ==========================================
# 17. Accuracy
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==========================================")
print("MODEL ACCURACY")
print("==========================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# 18. Classification Report
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4
    )
)


# ==========================================
# 19. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)


# ==========================================
# 20. Save Evaluation Results
# ==========================================

evaluation_results = {
    "accuracy": float(accuracy),

    "classification_report":
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        ),

    "confusion_matrix":
        cm.tolist()
}


with open(
    os.path.join(
        MODEL_DIR,
        "evaluation_results.json"
    ),
    "w"
) as file:

    json.dump(
        evaluation_results,
        file,
        indent=4
    )


print("\nEvaluation results saved to:")

print(
    "saved_model/evaluation_results.json"
)