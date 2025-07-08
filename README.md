# 🎙️ Speech-Based Emotion Detection using Fine-Tuned HuBERT

This project implements a speech emotion recognition system using a fine-tuned [HuBERT](https://arxiv.org/abs/2106.07447) model. The system was trained on an aggregated set of four widely used emotional audio datasets and achieves high accuracy for real-world applications like virtual assistants, sentiment-aware systems, and healthcare monitoring.

---

## 📌 Overview

Emotion recognition from speech is a key challenge in affective computing. Leveraging the power of **self-supervised learning**, we fine-tuned a **HuBERT** model on an aggregated dataset combining:

- SAVEE  
- TESS  
- CREMA-D  
- RAVDESS  

The model classifies speech into 6 emotion categories: `angry`, `happy`, `neutral`, `sad`, `fearful`, and `disgust`.

---

## 🗃️ Datasets

All audio datasets were resampled to 16kHz and preprocessed to remove noise. Emotion labels were normalized to 6 unified classes.

| Dataset   | Speakers | Emotions |
|-----------|----------|----------|
| SAVEE     | 4 (male) | 7        |
| TESS      | 2 (female)| 7        |
| CREMA-D   | 91       | 6        |
| RAVDESS   | 24       | 8        |

Dataset sources:
- [SAVEE](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)
- [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad)
- [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

---

## ⚙️ Methodology

### 1. Preprocessing
- Unified sample rate: 16kHz
- Emotion mapping to 6 classes
- Used `Wav2Vec2FeatureExtractor` from Hugging Face

### 2. Model: HuBERT
- Pretrained Model: [`superb/hubert-large-superb-er`](https://huggingface.co/superb/hubert-large-superb-er)
- Optimizer: AdamW
- Learning rate: `1e-5`
- Batch size: `2`
- Epochs: `10`

---

## 📊 Results

- **Training Accuracy**: 97%
- **Test Accuracy**: **99.37%**
- Model shows minimal misclassification, even across mixed-speaker datasets.

### Training Progress

| Epoch | Loss   | Accuracy |
|-------|--------|----------|
| 1     | 2.025  | 63.7%    |
| 5     | 0.000  | 93.1%    |
| 10    | 0.000  | **97.0%**|

---

## 📈 Confusion Matrix

The final confusion matrix shows near-perfect classification across all 6 emotion categories with minor overlap in closely related emotions (e.g., neutral vs sad).

---

## 🔍 Key Findings

- Transfer learning with HuBERT drastically reduces training time.
- Combining multiple datasets enhances generalizability.
- Works well across gender and accent variations.

---

## 🚀 Future Work

- Real-time emotion detection
- Multilingual emotion datasets
- Robustness to noisy environments

---

## 👥 Contributors

- **Dhrumil Patel** – [dpate371@lakeheadu.ca](mailto:dpate371@lakeheadu.ca)
- **Abhinav Pandey** – [apande11@lakeheadu.ca](mailto:apande11@lakeheadu.ca)

---

## 📚 References

1. [HuBERT Paper (2021)](https://arxiv.org/abs/2106.07447)  
2. [Hugging Face Model](https://huggingface.co/superb/hubert-large-superb-er)  
3. [Speech Emotion Recognition using CNN](https://ieeexplore.ieee.org/document/8706610)  
4. [MFCC + SVM for SER](https://ieeexplore.ieee.org/abstract/document/7877753)  

---

## 📝 License

This project is for academic and research use. Please cite appropriately if used in publications.
