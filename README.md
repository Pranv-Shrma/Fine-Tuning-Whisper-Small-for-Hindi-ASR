# Fine-Tuning Whisper Small for Hindi Speech Recognition

![Whisper Architecture](https://img.shields.io/badge/Model-Whisper%20Small-green) ![Language](https://img.shields.io/badge/Language-Hindi-orange) ![Compute](https://img.shields.io/badge/Compute-Google%20Colab%20T4-blueviolet) ![Metric](https://img.shields.io/badge/Metric-WER-blue)

This repository documents the process of fine-tuning OpenAI's **Whisper Small** model on a custom Hindi dataset. The project involves a robust data processing pipeline—converting raw audio URLs and JSON transcripts into segmented datasets—followed by fine-tuning and evaluation against standard benchmarks like FLEURS.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Compute Environment](#compute-environment)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Installation](#installation)
- [Training Configuration](#training-configuration)
- [Model Usage](#model-usage)
- [Evaluation Results](#evaluation-results)
- [Future Improvements](#future-improvements)

## 🚀 Project Overview

The goal of this project is to adapt the multilingual Whisper Small model specifically for Hindi Automatic Speech Recognition (ASR). Unlike standard training pipelines using pre-baked datasets, this project implements a custom ingestion engine that:
1.  Downloads audio and metadata from remote storage.
2.  Segments long audio files based on timestamped JSON transcriptions.
3.  Filters for valid Hindi characters (Devanagari script) to ensure data quality.
4.  Fine-tunes the model using the Hugging Face `Seq2SeqTrainer`.

**Model Hub:** [Pranav13/whisper-small-hi-custom-final-new](https://huggingface.co/Pranav13/whisper-small-hi-custom-final-new)

## 💻 Compute Environment

This project was executed efficiently using cloud-based resources provided by **Google Colab**.

*   **Platform:** Google Colab
*   **Hardware:** NVIDIA T4 GPU
*   **VRAM:** 16GB GPU RAM

The training batch sizes and gradient accumulation steps were specifically optimized to fit within the 16GB memory constraint of this environment while maintaining training stability.

## 🛠 Dataset & Preprocessing

The training data was derived from a custom dataset provided via CSV, containing URLs to raw `.wav` files and corresponding `.json` transcripts.

### The Pipeline
1.  **Ingestion:** Parsed a CSV containing `audio_url` and `transcription_url`.
2.  **Audio Segmentation:**
    *   Downloaded raw audio files.
    *   Parsed JSON transcripts to extract start/end times and speaker IDs.
    *   Clipped audio using `pydub` based on timestamps.
    *   Split segments exceeding 30 seconds to fit Whisper's input constraints.
3.  **Data Cleaning:**
    *   **Script Validation:** Implemented a regex filter (`[\u0900-\u097F]`) to ensure transcripts contained valid Devanagari characters, removing redacted or English-only segments.
    *   **Normalization:** Resampled all audio clips to **16kHz**.
4.  **Formatting:** Converted the processed segments into a Hugging Face `Dataset` object for training.

**Statistics:**
*   **Total Audio Files Processed:** 104
*   **Total Valid Segments:** ~5,700
*   **Total Invalid/Redacted Segments:** ~230

## 📦 Installation

To replicate this environment, install the required dependencies:

```bash
pip install uv
uv pip install transformers datasets torch librosa jiwer evaluate soundfile tensorboard gradio accelerate
```

## ⚙️ Training Configuration

The model was fine-tuned using the `Seq2SeqTrainer` with the following hyperparameters, optimized for the T4 GPU environment:

| Hyperparameter | Value | Notes |
| :--- | :--- | :--- |
| **Base Model** | `openai/whisper-small` | ~244M parameters |
| **Task** | Transcribe | |
| **Language** | Hindi | |
| **Learning Rate** | 1e-5 | |
| **Batch Size** | 8 (per device) | Adjusted for 16GB VRAM |
| **Gradient Accumulation** | 2 steps | Effective Batch Size: 16 |
| **Max Steps** | 2000 | |
| **Precision** | FP16 (Mixed Precision) | Faster training on T4 |
| **Warmup Steps** | 500 | |
| **Evaluation Strategy** | Every 500 steps | |

## 🎙 Model Usage

You can use the fine-tuned model for inference using the `pipeline` API from Transformers.

```python
import torch
from transformers import pipeline

# Load the fine-tuned model
model_id = "Pranav13/whisper-small-hi-custom-final-new"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    tokenizer=model_id,
    device=device
)

# proper audio file path (16kHz)
audio_path = "path/to/your/hindi_audio.wav"

# Transcribe
result = pipe(audio_path)
print(result["text"])
```

## 📊 Evaluation Results

The model was evaluated on the **FLEURS (Hindi)** test split to ensure it generalizes well to unseen data.

*   **Metric:** Word Error Rate (WER)
*   **Test Dataset:** `google/fleurs` (hi_in)
*   **Result:** **39.95%**

### Evaluation Code Snippet
```python
from datasets import load_dataset
import evaluate

# Load FLEURS Hindi test set
dataset = load_dataset("google/fleurs", "hi_in", split="test")
wer_metric = evaluate.load("wer")

# ... (Inference loop) ...

print(f"WER: {wer * 100:.2f}%")
```

## 🔮 Future Improvements

To further lower the WER and improve robustness:
1.  **Data Augmentation:** Apply noise injection and speed perturbation during training.
2.  **Larger Model:** Fine-tune `whisper-medium` or `whisper-large-v3` if compute allows (requires >16GB VRAM or PEFT/LoRA).
3.  **Hyperparameter Tuning:** Experiment with lower learning rates and cosine schedulers.
4.  **Language Model Decoding:** Integrate a Hindi N-gram language model during decoding to correct spelling inconsistencies.

---

## 👨‍💻 Author
**Pranav**
*   Email: reach.sharmapranav@gmail.com
