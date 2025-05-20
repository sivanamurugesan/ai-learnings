# ai-learnings
# 🤖 Red Hat Enterprise Linux AI

Learn **Red Hat Enterprise Linux (RHEL) AI** from scratch with **hands-on tutorials** and **enterprise-grade use-cases**. This repository guides you through the foundations of AI on RHEL and shows how to build, deploy, and optimize real-world AI models in hybrid cloud environments using Red Hat technologies.

---

## 🚀 Getting Started

This repo is organized into the following learning modules:

- 🔧 [Basic AI Terminologies](#-basic-ai-terminologies)
- 🧠 [Use Case 1: Model Training with PyTorch + DeepSpeed](#-use-case-1-model-training---pytorch--deepspeed)
- 🌐 [Use Case 2: Model Deployment with OpenShift AI](#-use-case-2-deployment---openshift-ai)
- ⚙️ [Use Case 3: AI Workload Optimization (NVIDIA, Intel, AMD)](#-use-case-3-optimization---hardware-acceleration)
- 📊 [Use Case 4: Generative AI Fine-Tuning with InstructLab](#-use-case-4-fine-tuning---instructlab)

---

## 📚 Basic AI Terminologies

| Term | Definition |
|------|------------|
| **Artificial Intelligence (AI)** | The simulation of human intelligence in machines that are programmed to think and learn. |
| **Machine Learning (ML)** | A subset of AI that enables machines to learn from data and improve over time without being explicitly programmed. |
| **Deep Learning** | A subset of ML that uses neural networks with many layers to analyze complex data patterns. |
| **Neural Network** | A computational model inspired by the human brain, consisting of layers of interconnected nodes (neurons). |
| **Natural Language Processing (NLP)** | A field of AI focused on enabling machines to understand, interpret, and respond to human language. |
| **Computer Vision** | Enables machines to interpret and make decisions based on visual data like images or videos. |
| **Generative AI** | AI systems that can create new content, such as text, images, or music, based on learned patterns. |
| **Cloud AI** | AI services and tools provided through cloud platforms like AWS, Azure, and Red Hat OpenShift AI. |
| **Hybrid Cloud** | Combines on-premises infrastructure with cloud services for flexibility and scalability. |
| **Inference** | Using a trained AI model to make predictions or decisions based on new data. |

---

# AI Terminologies with Analogies

| **Term**                     | **Definition**                                                                                     | **Analogy**                                                                                              |
|-----------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **Artificial Intelligence (AI)** | The simulation of human intelligence in machines that are programmed to think and learn.            | Like teaching a robot to act like a human — it can think, plan, and make decisions like a smart pet.     |
| **Machine Learning (ML)**       | A subset of AI that enables machines to learn from data and improve over time without being explicitly programmed. | Like training a child to recognize animals by showing pictures — the more examples, the better they get. |
| **Deep Learning**              | A subset of ML that uses neural networks with many layers to analyze complex data patterns.        | Like layers of filters in photo editing — each layer refines and adds detail to the final image.         |
| **Neural Network**             | A computational model inspired by the human brain, consisting of layers of interconnected nodes (neurons). | Like a network of pipes passing water (data) from one junction (neuron) to another, adjusting flow.       |
| **Natural Language Processing (NLP)** | A field of AI focused on enabling machines to understand, interpret, and respond to human language. | Like teaching a foreigner to speak your language and understand emotions in tone and words.              |
| **Computer Vision**            | Enables machines to interpret and make decisions based on visual data like images or videos.      | Like giving sight to a robot — now it can "see" and recognize faces, cars, or objects like a human does. |
| **Generative AI**              | AI systems that can create new content, such as text, images, or music, based on learned patterns. | Like an artist who learns thousands of styles and then creates original artwork from imagination.         |
| **Cloud AI**                   | AI services and tools provided through cloud platforms like AWS, Azure, and Red Hat OpenShift AI. | Like renting a powerful AI brain from the internet instead of building and maintaining your own.         |
| **Hybrid Cloud**               | Combines on-premises infrastructure with cloud services for flexibility and scalability.          | Like keeping valuables at home while renting extra space elsewhere — best of both worlds.                |
| **Inference**                  | Using a trained AI model to make predictions or decisions based on new data.                     | Like a doctor making a diagnosis based on past experience — the AI is applying what it has learned.      |

# What is an AI Model?

**AI models** (Artificial Intelligence models) are computational systems designed to simulate human intelligence. These models are trained using data to perform tasks such as recognizing patterns, making decisions, understanding language, and generating content.

---

## 🧠 Types of AI Models

### 1. **Machine Learning (ML) Models**
Models that learn from data to make predictions or decisions.

- **Supervised Learning** (uses labeled data):
  - Examples: Linear Regression, Decision Trees, Support Vector Machines (SVM), Random Forest
- **Unsupervised Learning** (uses unlabeled data):
  - Examples: K-Means Clustering, Principal Component Analysis (PCA)
- **Reinforcement Learning** (learns via rewards and penalties):
  - Examples: Deep Q-Network (DQN), Proximal Policy Optimization (PPO)

---

### 2. **Deep Learning Models**
A subset of ML using multi-layered neural networks.

- **Convolutional Neural Networks (CNNs)** – for images and video
- **Recurrent Neural Networks (RNNs), LSTMs** – for sequences like text or time series
- **Transformers** – advanced models for language tasks (e.g., GPT, BERT)

---

### 3. **Natural Language Processing (NLP) Models**
Focus on understanding and generating human language.

- Examples: GPT (Generative Pre-trained Transformer), BERT, T5

---

### 4. **Generative Models**
Designed to generate new data that mimics real data.

- **GANs (Generative Adversarial Networks)** – generate realistic images
- **Diffusion Models** – used for high-quality image generation (e.g., DALL·E)
- **LLMs (Large Language Models)** – generate human-like text (e.g., ChatGPT)

---

## 🔑 Key Concepts

- **Training**: Teaching the model using data
- **Inference**: Using the trained model to make predictions
- **Parameters**: Weights and biases the model learns
- **Data**: Essential fuel for training AI models

---

## 💡 Example: ChatGPT

**ChatGPT** is a **Large Language Model (LLM)** based on the **Transformer** architecture. It was trained on a vast dataset of text to understand and generate human-like language, useful for:
- Answering questions
- Writing content
- Translating languages
- And more

---


# Data Types in AI Models

AI models process and compute data in different **data types**, which can be categorized into:

1. **Semantic Data Types** – Types of data AI models work with (e.g., text, images).
2. **Computational Data Types** – Numerical precision formats used internally.

---

## 1. Semantic Data Types (High-Level Input Types)

These define the nature or kind of data used for training or inference.

| Data Type       | Description                            | Examples                          |
|------------------|----------------------------------------|-----------------------------------|
| **Numerical**     | Quantitative data (discrete/continuous) | Age, price, temperature           |
| **Categorical**   | Limited categories or labels            | Gender, country, color            |
| **Text (NLP)**    | Natural language input                  | Sentences, chat logs, reviews     |
| **Image**         | Visual data as pixel arrays             | Photos, X-rays, digits            |
| **Audio**         | Sound waveforms or spectrograms         | Speech, music                     |
| **Time Series**   | Sequential data indexed by time         | Stock prices, sensor data         |
| **Tabular**       | Structured rows/columns                 | CSV files, databases              |
| **Graph**         | Node-edge data structure                | Social networks, knowledge graphs |

---

## 2. Computational Data Types (Numeric Precision)

These define how the data is represented and computed within models, especially deep learning frameworks.

| Data Type      | Description                          | Use Cases                                       |
|----------------|--------------------------------------|------------------------------------------------|
| **int8 / uint8** | 8-bit integers                      | Quantized models for faster inference          |
| **float16 (FP16)** | 16-bit floating point              | Efficient training/inference, memory savings   |
| **bfloat16**    | Brain FP16 (used in TPUs)            | Better dynamic range than FP16                 |
| **float32 (FP32)** | 32-bit float (standard)           | Default in many ML/DL models                   |
| **float64 (FP64)** | 64-bit float                      | Scientific computing, rarely used in DL        |
| **bool**        | Boolean type                         | Logical conditions, masking                    |

---

## 3. Data Types in DL Frameworks (PyTorch, TensorFlow)

- Models operate on **tensors** (multi-dimensional arrays).
- Tensor data types include:
  - PyTorch: `torch.float32`, `torch.int8`, etc.
  - TensorFlow: `tf.float16`, `tf.bool`, etc.
- Precision affects:
  - **Speed**
  - **Memory usage**
  - **Model accuracy**

---

## 4. Why Data Types Matter

- **Model Performance**: Lower precision = faster and more memory-efficient.
- **Accuracy vs Efficiency**: FP16 is faster, but might reduce accuracy.
- **Hardware Support**: GPUs/TPUs optimize certain data types.
- **Preprocessing**: Proper encoding (e.g., categorical labels) ensures model compatibility.

---

## Summary

- **Semantic types** define what the data *is*.
- **Computational types** define how the data is *processed*.
- Correct use of data types leads to better **efficiency**, **accuracy**, and **scalability** of AI models.


# Supervised vs Unsupervised Data in AI

In machine learning, data is categorized based on whether it comes with labels (outputs) or not. This leads to two main types of learning paradigms:

---

## 1. Supervised Data

### ✅ Definition:
Supervised data consists of **input-output pairs**, where each input (feature) is associated with a known correct output (label).

### 📦 Structure:
```
(Input features) → (Known label/target)
```

### 📊 Examples:
| Input (X)             | Output (Y)                  |
|----------------------|-----------------------------|
| Image of a cat       | Label: "Cat"                |
| Email text           | Label: "Spam" or "Not spam" |
| House size, location | Label: House price          |

### 🧠 Used In:
- Classification (e.g., image recognition, spam detection)
- Regression (e.g., price prediction, stock forecasting)

### 🎯 Goal:
Train a model to **predict labels** for new, unseen data.

---

## 2. Unsupervised Data

### ❌ Definition:
Unsupervised data consists of **inputs only**, with **no labeled outputs**. The model learns patterns or structure from the data itself.

### 📦 Structure:
```
(Input features) → (No labels)
```

### 📊 Examples:
| Input (X)                  |
|---------------------------|
| Customer purchase behavior|
| Web user clickstreams     |
| Gene expression data      |

### 🧠 Used In:
- Clustering (e.g., customer segmentation)
- Dimensionality reduction (e.g., PCA, t-SNE)
- Anomaly detection (e.g., fraud detection)

### 🎯 Goal:
Find **patterns, groups, or structure** in the data without prior labeling.

---

## 🔍 Comparison Table

| Feature              | Supervised Learning       | Unsupervised Learning     |
|----------------------|---------------------------|----------------------------|
| Labels               | Present (known outputs)    | Absent (no known outputs)  |
| Goal                 | Predict outcome            | Discover patterns          |
| Examples             | Classification, Regression | Clustering, PCA            |
| Training Data Format | (X, Y) pairs               | X only                     |

---

## 📝 Summary

- **Supervised data** is labeled and helps train models to predict outcomes.
- **Unsupervised data** is unlabeled and helps models uncover hidden structures or groupings.
- Choosing between them depends on **data availability** and the **problem type** you want to solve.




