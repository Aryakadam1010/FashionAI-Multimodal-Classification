# ==========================================
# Fashion Product Category Prediction
# Multimodal Deep Learning Project
#
# Image  : MobileNetV2
# Text   : DistilBERT
# Fusion : Concatenation
# Classifier : ANN
# ==========================================


# ==========================================
# 1. Import Libraries
# ==========================================

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from transformers import AutoTokenizer, AutoModel
import torch


# ==========================================
# 2. Configuration
# ==========================================

DATA_PATH = "dataset/styles.csv"
IMAGE_DIR = "dataset/images"

CLEANED_DATA_PATH = "dataset/cleaned_dataset.csv"
MODEL_DIR = "saved_model"

IMAGE_SIZE = (224, 224)
IMAGE_BATCH_SIZE = 32

TEXT_BATCH_SIZE = 16
MAX_LENGTH = 32

NUM_EPOCHS = 10

RANDOM_STATE = 42

TEXT_MODEL_NAME = "distilbert-base-uncased"


# ==========================================
# 3. Create Model Directory
# ==========================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# 4. Load Dataset
# ==========================================

print("\n==========================================")
print("LOADING DATASET")
print("==========================================")

df = pd.read_csv(
    DATA_PATH,
    engine="python",
    on_bad_lines="skip"
)

print("Original Dataset Shape:")
print(df.shape)


# ==========================================
# 5. Keep Required Columns
# ==========================================

df = df[
    [
        "id",
        "productDisplayName",
        "articleType"
    ]
]


# ==========================================
# 6. Remove Missing Values
# ==========================================

df = df.dropna().copy()

print("\nDataset Shape After Cleaning:")
print(df.shape)


# ==========================================
# 7. Select Top 10 Article Types
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


print("\nSelected Article Types:")
print(df["articleType"].value_counts())


# ==========================================
# 8. Create Image Paths
# ==========================================

df["image_path"] = (
    df["id"].astype(str) + ".jpg"
)

df["image_path"] = df["image_path"].apply(
    lambda x: os.path.join(IMAGE_DIR, x)
)


# ==========================================
# 9. Keep Only Existing Images
# ==========================================

df = df[
    df["image_path"].apply(os.path.exists)
].copy()


print("\nImages Found:")
print(len(df))


# ==========================================
# 10. Encode Labels
# ==========================================

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(
    df["articleType"]
)

NUM_CLASSES = len(label_encoder.classes_)

print("\nLabel Mapping:")

for index, category in enumerate(label_encoder.classes_):
    print(index, "->", category)


# ==========================================
# 11. Save Cleaned Dataset
# ==========================================

df.to_csv(
    CLEANED_DATA_PATH,
    index=False
)

print(
    "\nCleaned Dataset Saved:",
    CLEANED_DATA_PATH
)


# ==========================================
# 12. Train / Validation / Test Split
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


print("\n==========================================")
print("DATASET SPLIT")
print("==========================================")

print("Training Samples   :", len(train_df))
print("Validation Samples :", len(val_df))
print("Testing Samples    :", len(test_df))


# ==========================================
# 13. Save Label Mapping
# ==========================================

label_mapping = {
    str(index): category
    for index, category
    in enumerate(label_encoder.classes_)
}

with open(
    os.path.join(MODEL_DIR, "label_mapping.json"),
    "w"
) as file:

    json.dump(
        label_mapping,
        file,
        indent=4
    )


# ==========================================
# 14. MobileNetV2 Feature Extractor
# ==========================================

print("\n==========================================")
print("LOADING MOBILENETV2")
print("==========================================")


image_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

# Freeze pretrained CNN
image_model.trainable = False


print("MobileNetV2 Loaded")
print("Input Shape :", image_model.input_shape)
print("Output Shape:", image_model.output_shape)
print("Trainable   :", image_model.trainable)


# ==========================================
# 15. Image Loading Function
# ==========================================

def load_image(image_path):
    """
    Loads one image and prepares it for MobileNetV2.
    """

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(image, dtype=np.float32)

    image = preprocess_input(image)

    return image


# ==========================================
# 16. Extract Image Features
# ==========================================

def extract_image_features(dataframe, split_name):
    """
    Extracts MobileNetV2 features for all images.
    """

    features = []

    image_paths = dataframe["image_path"].tolist()

    total = len(image_paths)

    print(
        f"\nExtracting image features for {split_name}..."
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

            try:

                image = load_image(path)

                batch_images.append(image)

            except Exception as error:

                print(
                    "\nImage error:",
                    path,
                    error
                )

                # Use a zero image if loading fails
                batch_images.append(
                    np.zeros(
                        (
                            IMAGE_SIZE[0],
                            IMAGE_SIZE[1],
                            3
                        ),
                        dtype=np.float32
                    )
                )

        batch_images = np.array(
            batch_images,
            dtype=np.float32
        )

        batch_features = image_model.predict(
            batch_images,
            verbose=0
        )

        features.append(batch_features)

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
# 17. Load DistilBERT
# ==========================================

print("\n==========================================")
print("LOADING DISTILBERT")
print("==========================================")


tokenizer = AutoTokenizer.from_pretrained(
    TEXT_MODEL_NAME
)

text_model = AutoModel.from_pretrained(
    TEXT_MODEL_NAME
)

# Freeze DistilBERT
text_model.eval()

for parameter in text_model.parameters():
    parameter.requires_grad = False


# ==========================================
# 18. Select Device
# ==========================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


text_model = text_model.to(device)


print("DistilBERT Loaded")
print("Device:", device)


# ==========================================
# 19. Extract Text Features
# ==========================================

def extract_text_features(dataframe, split_name):
    """
    Converts product names into DistilBERT feature vectors.
    """

    features = []

    texts = (
        dataframe["productDisplayName"]
        .astype(str)
        .tolist()
    )

    total = len(texts)

    print(
        f"\nExtracting text features for {split_name}..."
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

        # last_hidden_state:
        # [batch, sequence_length, hidden_size]

        hidden_states = (
            outputs.last_hidden_state
        )

        # Attention-mask mean pooling
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
            summed / counts.clamp(min=1e-9)
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
# 20. Extract Features
# ==========================================

print("\n==========================================")
print("FEATURE EXTRACTION")
print("==========================================")


X_train_image = extract_image_features(
    train_df,
    "TRAIN IMAGES"
)

X_val_image = extract_image_features(
    val_df,
    "VALIDATION IMAGES"
)

X_test_image = extract_image_features(
    test_df,
    "TEST IMAGES"
)


X_train_text = extract_text_features(
    train_df,
    "TRAIN TEXT"
)

X_val_text = extract_text_features(
    val_df,
    "VALIDATION TEXT"
)

X_test_text = extract_text_features(
    test_df,
    "TEST TEXT"
)


# ==========================================
# 21. Display Feature Shapes
# ==========================================

print("\n==========================================")
print("FEATURE SHAPES")
print("==========================================")

print(
    "Train Image Features:",
    X_train_image.shape
)

print(
    "Train Text Features :",
    X_train_text.shape
)

print(
    "Validation Image Features:",
    X_val_image.shape
)

print(
    "Validation Text Features :",
    X_val_text.shape
)

print(
    "Test Image Features:",
    X_test_image.shape
)

print(
    "Test Text Features :",
    X_test_text.shape
)


# ==========================================
# 22. Feature Fusion
# ==========================================

print("\n==========================================")
print("FEATURE FUSION")
print("==========================================")


X_train = np.concatenate(
    [
        X_train_image,
        X_train_text
    ],
    axis=1
)

X_val = np.concatenate(
    [
        X_val_image,
        X_val_text
    ],
    axis=1
)

X_test = np.concatenate(
    [
        X_test_image,
        X_test_text
    ],
    axis=1
)


print(
    "Fused Training Features:",
    X_train.shape
)

print(
    "Fused Validation Features:",
    X_val.shape
)

print(
    "Fused Testing Features:",
    X_test.shape
)


# ==========================================
# 23. Prepare Labels
# ==========================================

y_train = train_df["label"].values

y_val = val_df["label"].values

y_test = test_df["label"].values


# ==========================================
# 24. Build ANN Classifier
# ==========================================

print("\n==========================================")
print("BUILDING ANN CLASSIFIER")
print("==========================================")


input_size = X_train.shape[1]


inputs = layers.Input(
    shape=(input_size,),
    name="fused_features"
)


x = layers.Dense(
    512,
    activation="relu"
)(inputs)


x = layers.Dropout(
    0.30
)(x)


x = layers.Dense(
    256,
    activation="relu"
)(x)


x = layers.Dropout(
    0.20
)(x)


x = layers.Dense(
    128,
    activation="relu"
)(x)


outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="article_type"
)(x)


classifier = Model(
    inputs=inputs,
    outputs=outputs
)


# ==========================================
# 25. Compile Classifier
# ==========================================

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


classifier.summary()


# ==========================================
# 26. Train ANN
# ==========================================

print("\n==========================================")
print("TRAINING MULTIMODAL CLASSIFIER")
print("==========================================")


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


history = classifier.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=NUM_EPOCHS,

    batch_size=64,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ==========================================
# 27. Evaluate Model
# ==========================================

print("\n==========================================")
print("FINAL TEST EVALUATION")
print("==========================================")


test_loss, test_accuracy = classifier.evaluate(
    X_test,
    y_test,
    verbose=1
)


print(
    f"\nTest Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ==========================================
# 28. Save Classifier
# ==========================================

classifier_path = os.path.join(
    MODEL_DIR,
    "multimodal_classifier.keras"
)


classifier.save(
    classifier_path
)


print(
    "\nClassifier Saved:"
)

print(
    classifier_path
)


# ==========================================
# 29. Save Feature Extractor Information
# ==========================================

feature_info = {

    "image_model": "MobileNetV2",

    "image_input_size": [
        224,
        224,
        3
    ],

    "image_feature_dimension":
        int(X_train_image.shape[1]),

    "text_model":
        TEXT_MODEL_NAME,

    "text_feature_dimension":
        int(X_train_text.shape[1]),

    "fused_feature_dimension":
        int(X_train.shape[1]),

    "num_classes":
        NUM_CLASSES,

    "max_text_length":
        MAX_LENGTH

}


with open(
    os.path.join(
        MODEL_DIR,
        "feature_info.json"
    ),
    "w"
) as file:

    json.dump(
        feature_info,
        file,
        indent=4
    )


# ==========================================
# 30. Final Summary
# ==========================================

print("\n==========================================")
print("PROJECT TRAINING COMPLETE")
print("==========================================")

print(
    "Image Model      : MobileNetV2"
)

print(
    "Text Model       : DistilBERT"
)

print(
    "Fusion           : Feature Concatenation"
)

print(
    "Classifier       : ANN"
)

print(
    "Number of Classes:",
    NUM_CLASSES
)

print(
    f"Test Accuracy    : {test_accuracy * 100:.2f}%"
)

print(
    "\nSaved Files:"
)

print(
    "- saved_model/multimodal_classifier.keras"
)

print(
    "- saved_model/label_mapping.json"
)

print(
    "- saved_model/feature_info.json"
)