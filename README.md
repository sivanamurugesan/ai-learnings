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


# AWS AI Services and Use Cases

## 🧠 Core AWS AI Services and Use Cases

### 1. Amazon SageMaker
- **Purpose**: A comprehensive platform for building, training, and deploying machine learning models.
- **Use Cases**:
  - Predictive analytics (e.g., sales forecasting)
  - Anomaly detection in data
  - Custom model development for specific business needs
- **Analogy**: Think of it as a laboratory where data scientists experiment with different algorithms to find the best solution.

### 2. Amazon Rekognition
- **Purpose**: Provides image and video analysis capabilities.
- **Use Cases**:
  - Facial recognition for security
  - Object and scene detection in images and videos
  - Text detection in images
- **Analogy**: It's like giving your application the ability to see and interpret visual content.

### 3. Amazon Lex
- **Purpose**: A service for building conversational interfaces using voice and text.
- **Use Cases**:
  - Customer service chatbots
  - Voice-enabled applications
  - Automated response systems
- **Analogy**: Imagine a virtual assistant capable of understanding and responding to user queries.

### 4. Amazon Polly
- **Purpose**: Converts text into lifelike speech.
- **Use Cases**:
  - Voice-enabled applications
  - Audiobook creation
  - Language learning tools
- **Analogy**: It's like giving your application the ability to speak naturally.

### 5. Amazon Kinesis
- **Purpose**: A platform for real-time data streaming and analytics.
- **Use Cases**:
  - Real-time analytics on streaming data
  - Monitoring and processing log data
  - Real-time data ingestion for machine learning models
- **Analogy**: Think of it as a pipeline that transports data in real-time for immediate processing.

---

## 🚀 Generative AI on AWS

Generative AI refers to models that can generate new content, such as text, images, or music, based on learned patterns. AWS facilitates the development and deployment of generative AI applications through various services:

### Amazon Bedrock
- **Purpose**: A fully managed service that provides access to high-performing foundation models (FMs) from leading AI companies.
- **Use Cases**:
  - Building and scaling generative AI applications
  - Experimenting with multiple FMs to find the best fit for your use case
  - Fine-tuning models using your data without managing infrastructure
- **Analogy**: It's like a toolkit that provides various pre-built models, allowing you to choose the best one for your specific needs.

### Amazon Q
- **Purpose**: A powerful AI enterprise tool that integrates with various applications such as Gmail, Salesforce, and Zendesk.
- **Use Cases**:
  - Building applications using natural language commands
  - Automating tasks and workflows
  - Detecting security flaws in code
- **Analogy**: Think of it as a virtual assistant that can understand and execute tasks based on your instructions.

---

## 🛠️ Analogies to Understand AWS AI Services

- **Amazon SageMaker**: Comparable to a workshop where various tools and techniques are applied to solve complex problems.
- **Amazon Rekognition**: Similar to equipping an application with eyes to perceive and understand visual content.
- **Amazon Lex**: Acting as a virtual assistant capable of handling tasks and interactions autonomously.
- **Amazon Polly**: Functioning like a voice that can speak naturally and clearly.
- **Amazon Kinesis**: Serving as a conduit that transports data in real-time for immediate processing.
- **Amazon Bedrock**: Providing a toolkit with various pre-built models to choose from for specific needs.
- **Amazon Q**: Serving as a virtual assistant that can understand and execute tasks based on instructions.

---

# Azure AI Services and Use Cases

## 🧠 Core Azure AI Services and Use Cases

### 1. Azure OpenAI Service
- **Purpose**: Provides access to OpenAI's powerful language models, such as GPT-4, for generating human-like text.
- **Use Cases**:
  - Automated content generation (e.g., articles, summaries)
  - Customer support chatbots
  - Code generation and debugging
- **Analogy**: Think of it as a highly skilled writer or assistant that can understand and generate text based on prompts.

### 2. Azure Machine Learning
- **Purpose**: A cloud-based environment for training, deploying, and managing machine learning models.
- **Use Cases**:
  - Predictive analytics (e.g., sales forecasting)
  - Anomaly detection in data
  - Custom model development for specific business needs
- **Analogy**: It's like a laboratory where data scientists experiment with different algorithms to find the best solution.

### 3. Azure Cognitive Services
- **Purpose**: A collection of APIs that enable applications to see, hear, speak, understand, and interpret user needs.
- **Categories**:
  - **Vision**: Computer Vision, Custom Vision, Face API
  - **Speech**: Speech Recognition, Speech Synthesis
  - **Language**: Text Analytics, Translator, Language Understanding (LUIS)
  - **Decision**: Personalizer
- **Use Cases**:
  - Image and video analysis
  - Real-time translation and transcription
  - Sentiment analysis from text
- **Analogy**: These are like the sensory organs of an application, enabling it to perceive and understand the world.

### 4. Azure AI Search
- **Purpose**: A cloud search service with built-in AI capabilities to extract insights from content.
- **Use Cases**:
  - Enhanced search experiences in websites and applications
  - Document indexing and retrieval
- **Analogy**: Imagine a librarian who not only organizes books but also understands their content to provide more insightful recommendations.

### 5. Azure Bot Services
- **Purpose**: A platform for building and deploying intelligent bots.
- **Use Cases**:
  - Customer service automation
  - Appointment scheduling
  - Interactive FAQs
- **Analogy**: Think of it as a virtual receptionist who can handle inquiries and tasks without human intervention.

---

## 🚀 Generative AI on Azure

Generative AI refers to models that can generate new content, such as text, images, or music, based on learned patterns. Azure facilitates the development and deployment of generative AI applications through various services:

### Azure AI Foundry
- **Purpose**: A platform for training, testing, and deploying custom machine learning models, particularly useful for building Retrieval-Augmented Generation (RAG) systems.
- **Use Cases**:
  - Creating custom generative AI models
  - Experimenting with AI-driven content generation
  - Enhancing customer experiences through personalized content
- **Analogy**: It's like a workshop where you can craft and test new AI tools for your specific needs.

### Azure AI Studio
- **Purpose**: A low-code environment for developing AI applications, enabling rapid prototyping and deployment.
- **Use Cases**:
  - Building AI-powered applications quickly
  - Prototyping ideas without heavy coding
  - Iterating on AI models with ease
- **Analogy**: Think of it like a design studio where you sketch and build ideas quickly, with minimal technical barriers.

### Vector Databases in Azure
- **Purpose**: Azure provides efficient storage solutions for large datasets that power AI models, particularly generative AI.
- **Databases**:
  - **Azure Cosmos DB**: A globally distributed database service ideal for storing vast amounts of structured and unstructured data.
  - **Azure Cache for Redis**: An in-memory data store that reduces latency and improves performance.
  - **Azure Database for PostgreSQL - Flexible Server**: A managed database service supporting dynamic needs of AI applications.
- **Use Cases**:
  - Storing and querying large datasets for generative AI models
  - Enhancing performance with low-latency data access
  - Handling AI workloads efficiently
- **Analogy**: These are like powerful storage vaults, designed to hold and retrieve vast amounts of information quickly and efficiently.

---

## 🛠️ Analogies to Understand Azure AI Services

- **Azure OpenAI Service**: Like having a conversation with a knowledgeable assistant who can generate human-like text.
- **Azure Machine Learning**: Comparable to a workshop where various tools and techniques are applied to solve complex problems.
- **Azure Cognitive Services**: Similar to equipping an application with senses to perceive and understand its environment.
- **Azure AI Search**: Functioning like a smart librarian who not only organizes but also understands content to provide better search results.
- **Azure Bot Services**: Acting as a virtual agent capable of handling tasks and interactions autonomously.
- **Azure AI Foundry**: A creative workshop for designing and testing new AI models.
- **Azure AI Studio**: Like a design studio for building and experimenting with AI-powered applications.




