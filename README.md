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

## 🧠 Use Case 1: Model Training - PyTorch + DeepSpeed

### 🔍 Objective
Train large language models (LLMs) using RHEL-based infrastructure with PyTorch and DeepSpeed.

### ✅ Prerequisites
- RHEL 9+
- Python 3.9+
- CUDA drivers (if using GPU)
- PyTorch
- DeepSpeed

### 📘 Tutorial
```bash
sudo dnf install python3-pip git
pip3 install torch deepspeed transformers datasets

# Clone an example model
git clone https://github.com/huggingface/transformers
cd transformers/examples/pytorch/language-modeling

# Run training with DeepSpeed
deepspeed run_clm.py \
  --model_name_or_path gpt2 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --do_train \
  --output_dir ./gpt2-finetuned \
  --deepspeed ds_config.json

