
# Semantic Textual Similarity API - DataNeuron.ai Assignment

This repository contains a complete solution for the **Semantic Textual Similarity** task provided by DataNeuron.ai. The goal was to design an API that receives two text paragraphs and returns a similarity score ranging from 0 to 1 based on semantic similarity.

## 🔍 Problem Overview

Given a dataset with paragraph pairs, the task was to:

1. Build an algorithm to predict a similarity score between the paragraphs (Part A).
2. Deploy this algorithm as a cloud-hosted API endpoint (Part B).

## ✅ Key Features

- **Model Used**: [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **Framework**: Flask
- **Deployment**: Azure Web App with CI/CD pipeline using GitHub and Azure Container Registry
- **Token Limit Evaluation**:
  - `Longformer`: 4096 tokens (overhead, slower for smaller input sizes)
  - `all-MiniLM-L6-v2`: 512 tokens (faster, less accurate)
  - `all-mpnet-base-v2`: 512 tokens (best performance for ~400-word paragraphs)

### ⚖️ Model Choice Justification

After exploratory data analysis (EDA), the **average word count per paragraph** was found to be around **395 words**. Hence, `all-mpnet-base-v2`, known for better semantic alignment over medium to long-form texts, was chosen over lighter or excessively large models.

## 🐳 Deployment Overview

- **Dockerfile** was created to containerize the application.
- The image was pushed to **Azure Container Registry (ACR)**.
- The container was deployed to **Azure Web App** directly from ACR.
- Continuous deployment (CI/CD) was enabled via GitHub repository integration.

## 📡 API Specification

- **POST Endpoint**: `/`
- **Request**:
```json
{
  "text1": "First paragraph goes here...",
  "text2": "Second paragraph goes here..."
}
```
- **Response**:
```json
{
  "similarity_score": 0.84
}
```

## 🧪 Testing & Results

- The API returns a **cosine similarity score** clipped between `0` and `1`.
- UI available for testing via HTML form.
- Performance validated on ~3000 samples with high inference consistency.

## 📁 Project Structure

```
.
├── app.py
├── Dockerfile
├── templates/
│   └── UI.html
├── requirements.txt
├── report.pdf
└── README.md
```

## 📄 Report

Includes problem statement, EDA insights, model benchmarking, deployment methodology, and reasoning behind architectural decisions.

## 📌 Author

- Noothana Prasanna
- Contact: [jnoothan@gmail.com](mailto:jnoothan@gmail.com)

---

> 🚀 Deployed Azure Web App: [LIVE ENDPOINT](https://dataneuron.azurewebsites.net/)
