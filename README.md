# 👗 FashionAI — Multimodal Fashion Product Classification

> A multimodal deep learning system that combines **Computer Vision and Natural Language Processing** to automatically classify fashion products using both their images and textual descriptions.

---

## 📌 Overview

Fashion e-commerce platforms contain thousands of products that need to be accurately categorized for catalog management, search, filtering, inventory organization, and recommendation systems.

Traditional approaches may rely on either visual information or textual metadata.

**FashionAI** takes a multimodal approach by combining:

- 🖼️ Product Images
- 📝 Product Names

The system extracts meaningful representations from both modalities and combines them before performing the final classification.

### Core Architecture

```text
                         FASHION PRODUCT
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
              IMAGE                        TEXT
                 │                           │
                 ▼                           ▼
           MobileNetV2                  DistilBERT
                 │                           │
                 ▼                           ▼
        Visual Features              Text Features
            1280-D                        768-D
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               ▼
                       FEATURE FUSION
                          2048-D
                               │
                               ▼
                         ANN CLASSIFIER
                               │
                               ▼
                     FASHION CATEGORY
