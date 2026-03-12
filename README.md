# PII Detection using Multilingual BERT

This project builds a **Named Entity Recognition (NER)** model for detecting **Personally Identifiable Information (PII)** using **Multilingual BERT (mBERT)**.

The model is trained on the **AI4Privacy PII Masking dataset** and learns to identify sensitive information such as phone numbers, emails, addresses, credit card numbers, and more.

The goal is to automatically **detect and mask sensitive information in text**.

---

# Dataset

The model uses the **ai4privacy/pii-masking-200k** dataset from HuggingFace.

Dataset characteristics:

* 209,261 training samples
* Multilingual text
* Pre-tokenized for mBERT
* BIO tagging format

Important fields in the dataset:

| Field             | Description                 |
| ----------------- | --------------------------- |
| source_text       | Original sentence           |
| target_text       | Sentence with masked PII    |
| privacy_mask      | Exact span locations of PII |
| mbert_text_tokens | Tokenized input             |
| mbert_bio_labels  | BIO labels for tokens       |
| language          | Language of text            |

Example:

**Source**

A student's assessment was found on device bearing IMEI: 06-184755-866851-3.

**Target**

A student's assessment was found on device bearing IMEI: [PHONEIMEI].

---

# Model

The model used is:

```
bert-base-multilingual-cased
```

Architecture:

```
Input Text
     │
Tokenizer (mBERT)
     │
Multilingual BERT Encoder
     │
Token Classification Head
     │
BIO Label Prediction
```

Each token receives a label such as:

```
O
B-EMAIL
I-EMAIL
B-PHONENUMBER
I-PHONENUMBER
B-CREDITCARDNUMBER
```

---

# Project Pipeline

## 1. Load Dataset

The dataset is loaded using HuggingFace datasets.

```
from datasets import load_dataset

raw_dataset = load_dataset("ai4privacy/pii-masking-200k")
```

---

## 2. Extract Labels

All BIO labels are collected and converted to numerical IDs.

```
label_set = set()

for labels in train_ds['mbert_bio_labels']:
    label_set.update(labels)

label_list = sorted(label_set)
```

Label mappings:

```
label2id = {label: idx}
id2label = {idx: label}
```

---

## 3. Encode Labels

BIO labels are converted into numerical tags for training.

```
def encode_labels(example):
    example['ner_tags'] = [label2id[i] for i in example['mbert_bio_labels']]
    return example
```

---

## 4. Train / Test Split

Dataset is split into training and testing sets.

* 80% Training
* 20% Testing

---

## 5. Model Initialization

```
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)
```

---

## 6. Device Setup

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

---

# Example Inference

Example input:

Hello, my name is Sarah Johnson.
I live at 42 Maple Street, New York,
and my email is [sarah.j@gmail.com](mailto:sarah.j@gmail.com).

Token prediction output:

```
Sarah                 → B-FIRSTNAME
Johnson               → I-FIRSTNAME
sarah.j@gmail.com     → B-EMAIL
```

The model identifies sensitive entities **token-by-token**.

---

# Installation

Clone the repository:

```
git clone https://github.com/sanjeevanii/PrivacyBERT.git
cd PrivacyBERT
```

Install dependencies:

```
pip install torch
pip install transformers
pip install datasets
pip install evaluate
pip install accelerate
pip install matplotlib
pip install pandas
pip install numpy
```

---

# Running the Notebook

Open the notebook:

```
jupyter notebook
```

Run all cells to:

1. Load dataset
2. Preprocess labels
3. Initialize model
4. Run inference

---

# Dependencies

Main libraries used:

* PyTorch
* HuggingFace Transformers
* HuggingFace Datasets
* Accelerate
* Evaluate
* NumPy
* Pandas
* Matplotlib

---

# Applications

This system can be used for:

* Privacy-preserving NLP
* Data anonymization
* Log sanitization
* Compliance (GDPR / HIPAA)
* Secure document processing
