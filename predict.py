# ==========================================
# Fashion Product Category Prediction
# Prediction Script
# ==========================================

import os
import json
import numpy as np
import tensorflow as tf
import torch

from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from transformers import AutoTokenizer, AutoModel


# ==========================================
# 1. Configuration
# ==========================================

MODEL_DIR = "saved_model"

IMAGE_SIZE = (224, 224)

MAX_LENGTH = 32

TEXT_MODEL_NAME = "distilbert-base-uncased"

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR,
    "multimodal_classifier.keras"
)

LABEL_MAPPING_PATH = os.path.join(
    MODEL_DIR,
    "label_mapping.json"
)


# ==========================================
# 2. Load Saved Classifier
# ==========================================

print("Loading multimodal classifier...")

classifier = tf.keras.models.load_model(
    CLASSIFIER_PATH
)

print("Classifier loaded successfully.")


# ==========================================
# 3. Load Label Mapping
# ==========================================

with open(
    LABEL_MAPPING_PATH,
    "r"
) as file:

    label_mapping = json.load(file)


# ==========================================
# 4. Load MobileNetV2
# ==========================================

print("Loading MobileNetV2...")

image_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

image_model.trainable = False

print("MobileNetV2 loaded.")


# ==========================================
# 5. Load DistilBERT
# ==========================================

print("Loading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    TEXT_MODEL_NAME
)

text_model = AutoModel.from_pretrained(
    TEXT_MODEL_NAME
)

text_model.eval()


# ==========================================
# 6. Select Device
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
# 7. Image Feature Extraction
# ==========================================

def extract_image_feature(image_path):

    image = Image.open(
        image_path
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

    image = np.expand_dims(
        image,
        axis=0
    )

    image_feature = image_model.predict(
        image,
        verbose=0
    )

    return image_feature


# ==========================================
# 8. Text Feature Extraction
# ==========================================

def extract_text_feature(product_name):

    encoded = tokenizer(
        [product_name],
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

    text_feature = (
        summed /
        counts.clamp(min=1e-9)
    )

    text_feature = (
        text_feature
        .cpu()
        .numpy()
    )

    return text_feature


# ==========================================
# 9. Prediction Function
# ==========================================

def predict(image_path, product_name):

    # Image features
    image_feature = extract_image_feature(
        image_path
    )

    # Text features
    text_feature = extract_text_feature(
        product_name
    )

    # Fuse features
    fused_features = np.concatenate(
        [
            image_feature,
            text_feature
        ],
        axis=1
    )

    # Prediction
    probabilities = classifier.predict(
        fused_features,
        verbose=0
    )

    predicted_index = int(
        np.argmax(probabilities[0])
    )

    confidence = float(
        probabilities[0][predicted_index]
    )

    predicted_category = label_mapping[
        str(predicted_index)
    ]

    return predicted_category, confidence


# ==========================================
# 10. Test Prediction
# ==========================================

if __name__ == "__main__":

    image_path = "dataset/images/11527.jpg"

    product_name = (
        "United Colors of Benetton Men Olive Tshirts"
    )

    category, confidence = predict(
        image_path,
        product_name
    )

    print("\n==========================================")
    print("PREDICTION")
    print("==========================================")

    print(
        "Product Name :",
        product_name
    )

    print(
        "Prediction   :",
        category
    )

    print(
        "Confidence   :",
        f"{confidence * 100:.2f}%"
    )