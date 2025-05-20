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

Would you like a diagram or code example included in this Markdown too?

