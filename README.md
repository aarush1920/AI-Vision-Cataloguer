# VisionCatalog-AI 🛍️🤖
> **Automated Open-Vocabulary E-Commerce Product Cataloger & Studio Asset Generator**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**VisionCatalog-AI** is a modular, **100% offline-first multimodal AI pipeline** engineered to eliminate the manual overhead of product cataloging for small merchants and exporters. It takes raw, unedited smartphone product photos and transforms them into studio-grade visual assets and marketplace-compliant product listings (Amazon & Shopify) locally on standard consumer hardware—completely free and without cloud API dependencies.

---

## 🌟 Key Architecture & Modules

The pipeline is structured into four sequential processing stages:

```text
[Raw Smartphone Photo] 
       │
       ▼
 1. Studio Matting Engine ──> rembg (U²-Net) + OpenCV (Ground Contact Shadows, 2000x2000px White Canvas)
       │
       ▼
 2. Multimodal VLM Parser ──> Microsoft Florence-2-base (Dense Captioning, OCR Text & Attribute Extraction)
       │
       ▼
 3. Copywriting Engine    ──> Intelligent Domain Router & Universal Attribute-Driven Fallback System
       │
       ▼
 4. Streamlit Dashboard   ──> Human-in-the-Loop (HITL) Interactive UI with Batch CSV/ZIP Exports
