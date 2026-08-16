import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# ============================================================
# 1. LOAD FASHION-MNIST DATASET
# ============================================================

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

print("Original training shape:", x_train.shape)
print("Original test shape:", x_test.shape)


# ============================================================
# 2. NORMALIZATION
# ============================================================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# ============================================================
# 3. ADD CHANNEL DIMENSION
# ============================================================

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print("After preprocessing:")
print("Training shape:", x_train.shape)
print("Test shape:", x_test.shape)


# ============================================================
# 4. DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential([
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1
    )
])


# ============================================================
# 5. BUILD CNN MODEL
# ============================================================

model = keras.Sequential([

    # Input
    keras.layers.Input(shape=(28, 28, 1)),

    # Data Augmentation
    data_augmentation,

    # First Convolution Block
    keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Second Convolution Block
    keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Third Convolution Layer
    keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    # Convert feature maps to vector
    keras.layers.Flatten(),

    # Fully Connected Layer
    keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Output Layer
    keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# 6. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 7. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)


# ============================================================
# 9. EVALUATE MODEL ON TEST DATA
# ============================================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

predictions = model.predict(x_test)

predicted_classes = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# 11. CLASS NAMES
# ============================================================

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# ============================================================
# 12. DISPLAY ONE PREDICTION
# ============================================================

index = 0

predicted_class = np.argmax(
    predictions[index]
)

confidence = predictions[index][predicted_class]

print("\nSingle Prediction")
print("-----------------")
print("Predicted:", class_names[predicted_class])
print("Actual:", class_names[y_test[index]])
print("Confidence:", f"{confidence * 100:.2f}%")


plt.imshow(
    x_test[index].squeeze(),
    cmap="gray"
)

plt.title(
    f"Predicted: {class_names[predicted_class]} "
    f"({confidence * 100:.2f}%)\n"
    f"Actual: {class_names[y_test[index]]}"
)

plt.axis("off")
plt.show()


# ============================================================
# 13. FIND WRONG PREDICTIONS
# ============================================================

wrong_indices = np.where(
    predicted_classes != y_test
)[0]

print("\nNumber of wrong predictions:", len(wrong_indices))


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predicted_classes
)

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.xticks(
    range(10),
    class_names,
    rotation=45
)

plt.yticks(
    range(10),
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.title("Fashion-MNIST Confusion Matrix")

plt.colorbar()

plt.tight_layout()
plt.show()


# ============================================================
# 15. TRAINING ACCURACY vs VALIDATION ACCURACY
# ============================================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.show()


# ============================================================
# 16. TRAINING LOSS vs VALIDATION LOSS
# ============================================================

plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.show()