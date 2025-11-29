# Persian Sentiment Analysis with BERT (7 Emotions)

This repository provides code and data to **fine-tune a BERT model for Persian sentiment analysis**, classifying text into **7 emotion categories**:

- **HAPPY (0)**
- **SAD (1)**
- **FEAR (2)**
- **ANGRY (3)**
- **HATE (4)**
- **SURPRISE (5)**
- **OTHER (6)**

Uses pre-trained [HooshvareLab/bert-fa-base-uncased-sentiment-digikala](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-digikala) (Persian BERT on Digikala reviews), fine-tuned on custom Persian social media texts (tweets).

## Model Architecture (Post-Fine-Tuning)
- **Base**: BERT-base-uncased (768 hidden, 12 layers/heads, vocab 50k Persian)
- **Task**: Single-label classification (`BertForSequenceClassification`)
- **Train on**: Custom 7-class dataset
- **Output Dir**: `./model/` (generated after training)

## Dataset
- **train_clean.tsv**: Training set (~10k+ Persian texts, e.g., hate on birds, sad news)
- **test_clean.tsv**: Test set (similar structure)

Source: Cleaned Persian tweets/social media for multi-emotion analysis.

## Training (`codes/model_finetune.py`)
Fine-tune with Hugging Face `Trainer`:
- **Optimizer**: AdamW (lr=3e-5)
- **Epochs**: 3
- **Batch**: 16 (train)/64 (eval)
- **Max Len**: 128
- **Metrics**: Accuracy

**Run** (generates `./model/`):
```
pip install transformers datasets torch accelerate
python codes/model_finetune.py
```

## Inference (`codes/model_test.ipynb`)
After training, load pipeline:
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="./model", device=0)

texts = ["این فیلم فوق‌العاده بود!", "اصلاً خوشم نیومد.", "کمی نگرانم."]
print(classifier(texts))
```

**Labels** (id2label):
```
0: HAPPY, 1: SAD, 2: FEAR, 3: ANGRY, 4: HATE, 5: SURPRISE, 6: OTHER
```

## Quick Usage
1. **Setup**:
   ```
   pip install transformers torch datasets accelerate
   ```

2. **Train Model**:
   ```
   python codes/model_finetune.py
   ```

3. **Predict**:
   ```python
   classifier("متن فارسی")
   ```

## Repo Structure
```
.
├── README.md
├── codes/
│   ├── model_finetune.py  # Fine-tune script
│   └── model_test.ipynb   # Test notebook
└── dataset/
    ├── train_clean.tsv
    └── test_clean.tsv
```
*(Run training to generate `./model/`)*

## Results
Fine-tuned for nuanced Persian emotion detection (beyond pos/neg). Ideal for social media, feedback analysis.

## Credits
- [Hugging Face Transformers](https://huggingface.co/)
- [HooshvareLab Persian BERT](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-digikala)

Fork/star/contribute! 🚀
