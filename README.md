# FashionAI — Multimodal Fashion Product Classification

> **A multimodal deep learning system that combines product images and product names to automatically classify fashion products into 10 categories.**

<p align="center">

**MobileNetV2** • **DistilBERT** • **Feature Fusion** • **ANN** • **Flask**

</p>

---

## 🎯 Why Did I Build This?

Fashion e-commerce platforms contain thousands of products that need to be correctly categorized.

A traditional system might rely only on:

- Product images
- Product names
- Manually assigned categories

But a fashion product contains **multiple sources of information**.

For example:

> **"United Colors of Benetton Men Olive Tshirts"**

The **image** provides visual information, while the **product name** provides semantic information.

So instead of building an image-only classifier, I built a **multimodal deep learning system** that learns from both.

### The idea

```text
             Fashion Product
                   │
          ┌────────┴────────┐
          │                 │
        Image             Text
          │                 │
          ▼                 ▼
    MobileNetV2         DistilBERT
          │                 │
          ▼                 ▼
    1280 Features       768 Features
          │                 │
          └────────┬────────┘
                   │
                   ▼
            Feature Fusion
                   │
             2048 Features
                   │
                   ▼
              ANN Classifier
                   │
                   ▼
             Product Category
```

---

# 🚀 What Does The Project Do?

The application takes:

### 📷 Product Image

and

### 📝 Product Name

and predicts the product category.

### Example

**Input**

```text
Product Name:
United Colors of Benetton Men Olive Tshirts

Product Image:
[ Fashion Product Image ]
```

**Output**

```text
Predicted Category : Tshirts
Confidence         : 99.92%
```

---

# 🧠 What Did I Use?

| Component | Technology | Purpose |
|---|---|---|
| 🖼️ Image Model | **MobileNetV2** | Extract visual features |
| 📝 Text Model | **DistilBERT** | Extract semantic text features |
| 🔗 Fusion | **Feature Concatenation** | Combine image + text features |
| 🧠 Classifier | **ANN** | Predict final category |
| 🌐 Backend | **Flask** | Serve the ML model |
| 🎨 Frontend | **HTML + CSS** | User interface |
| 📊 Data Processing | **Pandas + NumPy** | Dataset preprocessing |
| 🔬 ML Utilities | **Scikit-learn** | Label encoding & data splitting |
| 👁️ Computer Vision | **OpenCV + Pillow** | Image processing |
| 🤗 NLP | **Hugging Face Transformers** | DistilBERT |
| 🧠 Deep Learning | **TensorFlow/Keras + PyTorch** | Model implementation |

---

# 🏗️ How The Model Works

The project has **4 major stages**.

### 1️⃣ Image Feature Extraction

I use pretrained **MobileNetV2** to extract visual features from the product image.

```text
Image
 ↓
Resize → 224 × 224
 ↓
MobileNetV2
 ↓
1280-dimensional image features
```

MobileNetV2 is used through **Transfer Learning** as a feature extractor.

---

### 2️⃣ Text Feature Extraction

The product name is processed using **DistilBERT**.

```text
Product Name
     ↓
Tokenizer
     ↓
DistilBERT
     ↓
768-dimensional text features
```

DistilBERT allows the system to capture semantic information from the product name.

---

### 3️⃣ Feature Fusion

The two feature vectors are concatenated.

```text
1280 Image Features
        +
768 Text Features
        │
        ▼
2048 Fused Features
```

---

### 4️⃣ Classification

The fused representation is passed into an Artificial Neural Network.

```text
2048
 │
 ▼
Dense(512)
 │
 ▼
Dropout
 │
 ▼
Dense(256)
 │
 ▼
Dropout
 │
 ▼
Dense(128)
 │
 ▼
Dense(10)
 │
 ▼
Fashion Category
```

---

# 📊 Dataset

The project uses a fashion product dataset containing product metadata and corresponding product images.

### Dataset Processing

```text
Original Dataset
44,424 records
       ↓
Data Cleaning
       ↓
Top 10 Categories
       ↓
Image Availability Check
       ↓
25,465 usable images
       ↓
Train / Validation / Test
```

### Dataset Split

| Dataset | Samples |
|---|---:|
| Training | 20,372 |
| Validation | 2,546 |
| Testing | 2,547 |

---

# 👕 Categories

The model currently predicts **10 fashion categories**:

```text
Casual Shoes
Handbags
Heels
Kurtas
Shirts
Sports Shoes
Sunglasses
Tops
Tshirts
Watches
```

---

# 📈 Model Performance

The multimodal classifier achieved:

## 🏆 98.00% Test Accuracy

```text
Image Model       : MobileNetV2
Text Model        : DistilBERT
Fusion            : Feature Concatenation
Classifier        : ANN
Classes           : 10

Test Loss         : 0.0568
Test Accuracy     : 98.00%
```

> The reported accuracy is on the held-out test split used during this project.

---

# 💡 Why Is This Interesting?

The main idea is not simply "CNN classification."

The project combines multiple AI concepts into one pipeline:

```text
Computer Vision
      +
Transfer Learning
      +
NLP
      +
Transformers
      +
Embeddings
      +
Multimodal Fusion
      +
Neural Networks
      +
Flask Deployment
```

The interesting part is the **multimodal architecture**.

Instead of asking:

> "What does this image look like?"

the system can use:

> **"What does this product look like + what does its name tell us?"**

---

# 🌍 Where Can This Be Used?

### 🛒 E-Commerce

Automatically categorize products when sellers upload them.

### 📦 Product Catalog Management

Automatically organize large product catalogs.

### 🏷️ Automatic Product Tagging

Generate category labels for newly uploaded products.

### 🔎 Product Search & Filtering

Automatically generated categories can improve product discovery.

### 🤖 AI-Assisted Product Upload

A seller can upload a product and receive an automatically suggested category.

### 🚀 Future Applications

The same multimodal architecture could later be extended to:

- Fashion recommendation
- Similar product search
- Visual product search
- Fashion attribute prediction
- Personalized recommendations

---

# 📁 Project Structure

```text
FashionAI-Multimodal-Classification/
│
├── dataset/
│   └── .gitkeep
│
├── saved_model/
│   ├── feature_info.json
│   ├── label_mapping.json
│   └── evaluation_results.json
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── inspect_dataset.py
│
├── cnn_basics.py
├── tensorflow_basics.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Important Files

| File | Purpose |
|---|---|
| `train.py` | Dataset preparation + model training |
| `predict.py` | Run predictions |
| `evaluate.py` | Evaluate the trained model |
| `app.py` | Flask web application |
| `templates/index.html` | Web interface |
| `requirements.txt` | Python dependencies |
| `saved_model/` | Model metadata & results |

---

# ⚙️ How To Run

## 1. Clone the Repository

```bash
git clone https://github.com/Aryakadam1010/FashionAI-Multimodal-Classification.git

cd FashionAI-Multimodal-Classification
```

---

## 2. Create Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Train The Model

If you want to train the model from scratch:

```bash
python train.py
```

The training pipeline will:

```text
Load Dataset
     ↓
Clean Data
     ↓
Select Categories
     ↓
Validate Images
     ↓
Encode Labels
     ↓
Split Dataset
     ↓
Load MobileNetV2
     ↓
Extract Image Features
     ↓
Load DistilBERT
     ↓
Extract Text Features
     ↓
Fuse Features
     ↓
Train ANN
     ↓
Evaluate Model
```

---

# 🔮 Run Prediction

```bash
python predict.py
```

This loads the trained components and predicts the category for a product.

---

# 🌐 Run The Web Application

Start Flask:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The application allows the user to:

1. Upload a product image
2. Enter the product name
3. Run the prediction
4. View the predicted category
5. View the confidence score

---

# 🔐 GitHub & Dataset

Large files such as:

- Dataset images
- Original CSV files
- Virtual environment
- Large trained model files

are excluded using `.gitignore`.

This keeps the GitHub repository lightweight and focused on the **source code and project architecture**.

---

# 🔮 Future Improvements

### 1. More Categories

Expand the classifier beyond the current 10 categories.

### 2. Fine-Tuning

Fine-tune MobileNetV2 and DistilBERT on fashion-specific data.

### 3. Better Fusion

Experiment with:

- Attention-based fusion
- Cross-modal attention
- Multimodal Transformers

### 4. Fashion Attributes

Predict:

```text
Color
Gender
Material
Pattern
Season
Occasion
```

### 5. Recommendation System

Use multimodal embeddings for fashion recommendations.

### 6. Similar Product Search

Find visually and semantically similar products.

---

# ⚠️ Important Project Scope

Despite the repository/project name **Product Authenticator**, the current implementation performs:

> **Fashion Product Category Classification**

It does **NOT** currently determine whether a product is genuine or counterfeit.

The current model predicts categories such as:

```text
Tshirts
Shirts
Watches
Shoes
Handbags
etc.
```

An actual **product authenticity/counterfeit detection system** would require a different dataset and modeling objective.

---

# 🏆 Project Summary

```text
                 Fashion Product
                        │
              ┌─────────┴─────────┐
              │                   │
            Image                Text
              │                   │
              ▼                   ▼
         MobileNetV2          DistilBERT
              │                   │
           1280                 768
              │                   │
              └─────────┬─────────┘
                        │
                  Feature Fusion
                        │
                     2048
                        │
                        ▼
                       ANN
                        │
                        ▼
                10 Fashion Classes
```

### Final Result

**98.00% Test Accuracy**

---

# 👨‍💻 Author

### Arya Kadam

Artificial Intelligence & Data Science

---

<p align="center">

⭐ If you found this project interesting, consider giving the repository a star!

</p>
