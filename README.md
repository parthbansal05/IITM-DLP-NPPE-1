# 🏛️ Ancient Texts Provenance Challenge

**Language Origin Detection using Fine-Tuned Large Language Models**  

---

## 📘 Overview

This project was developed as part of the **NPPE-1: Ancient Texts Provenance Challenge** hosted on **Kaggle**.  
The goal was to determine the **geographical origin of ancient inscriptions** based solely on textual content using modern **Large Language Models (LLMs)**.

The competition required participants to:

* Analyze cleaned, anonymized historical text data
* Fine-tune a multilingual LLM for classification
* Build a fully reproducible solution inside the Kaggle environment
* Optimize model performance for the **Macro F1-Score**

---

## 🧩 Competition Details

**Host:** Sherry Thomas  
**Platform:** [Kaggle Competition Link](https://www.kaggle.com/t/4673999fe3e84d14a17956a4d0c96432)  
**Evaluation Metric:** Macro F1-Score  
**Final Score:** `0.50185` (Public: `0.49734`)  
**Rank:** *22 out of 122 participants*  
**Maximum Score Achieved in Competition:** `0.56`  
**Participant:** *Parth Bansal (Team: 21F3000805)*  


### 🗓️ Timeline

| Event                | Date & Time (IST)    |
| -------------------- | -------------------- |
| Competition Start    | 17 Oct 2025, 5:00 PM |
| Submission Deadline  | 19 Oct 2025, 5:00 PM |
| Results Announcement | 19 Oct 2025, 7:00 PM |

---

## 📊 Dataset Overview

| File                      | Description                                    |
| ------------------------- | ---------------------------------------------- |
| **train.csv**             | 119,656 labeled inscriptions (id, text, label) |
| **test.csv**              | 29,914 unlabeled inscriptions (id, text)       |
| **sample_submission.csv** | Submission format (id, predicted label)        |

**Task:** Predict the anonymized geographical label for each inscription in the test set.

### ⚖️ Class Distribution

The dataset was **highly imbalanced**, with a skewed label distribution (e.g., Label 7 ≈ 30% of data, Label 14 < 0.01%).  
This imbalance required **class-weighted loss normalization** and **data resampling** to achieve stable performance.

---

## 🧠 Approach Overview

The project fine-tunes **XLM-RoBERTa Base**, a multilingual transformer model from Hugging Face, on the given dataset.  
The pipeline focuses on robust preprocessing, handling imbalance, and optimizing model generalization through weighted training.

---

## ⚙️ Implementation Steps

### **Step 1: Setup and Environment**

Configured Kaggle notebook environment with essential libraries:

```bash
pip install -U transformers huggingface_hub datasets accelerate scikit-learn
```

Disabled warnings, fixed random seeds, and ensured GPU utilization where available.

---

### **Step 2: Configuration**

Defined global hyperparameters via a `CFG` class:

* **Model:** `xlm-roberta-base`
* **Max sequence length:** 256
* **Batch size:** 16
* **Learning rate:** 2e-5
* **Epochs:** 3
* **Validation split:** 20%

---

### **Step 3: Data Loading and Preprocessing**

#### 3.1 Data Exploration

* Analyzed text length distribution and class imbalance
* Generated visualizations using Matplotlib & Seaborn

#### 3.2 Data Filtering

* Filtered samples by word length range (1–1200 words)
* Reduced outliers to stabilize training

#### 3.3 Data Segregation

* Applied **class-wise normalization** factors to balance the dataset — ensuring even distribution across classes without excessive over- or under-sampling, while preserving the original data characteristics
* Split into **train**, **validation**, and **internal test** sets using stratified sampling
* Converted to Hugging Face `DatasetDict` for efficient processing

---

### **Step 4: Tokenization**

* Used `AutoTokenizer` from Hugging Face for consistent preprocessing
* Applied truncation and dynamic padding using `DataCollatorWithPadding`
* Prepared tokenized datasets for all data splits

---

### **Step 5: Model**

#### 5.1 Class-Weighted Loss Normalization

* Calculated class weights using `compute_class_weight(balanced)`
* Applied **log normalization** (twice) on the initially computed `class_weight` values from `sklearn.utils.class_weight.compute_class_weight` to compress extreme outliers — especially large weights (e.g., `2868.53`) that could destabilize training.
* Followed with **min–max scaling (1.0–1.5 range)** to maintain proportional differences while constraining overall weight variation.
* This produced a **smooth bias gradient** across classes, allowing the model to learn balanced representations without distorting the original class distribution.
* Implemented **Weighted CrossEntropyLoss** to counter imbalance

#### 5.2 Custom Trainer Configuration

Created a subclass of `Trainer` to integrate class-weighted loss:

```python
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=scaled_weights_tensor)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

#### 5.3 Model Training

Trained `xlm-roberta-base` for 3 epochs with evaluation at each epoch using Macro F1.
Saved the best-performing checkpoint.

#### 5.4 Model Evaluation

Evaluated on the validation and internal test sets.
Metric used:

```python
f1_score(labels, predictions, average="macro")
```

---

### **Step 6: Prediction and Submission**

Generated predictions on the Kaggle test set:

```python
preds = trainer.predict(tokenized_ds['competition_test'])
```

Formatted results to match `sample_submission.csv`:

```python
submission_df['label'] = preds
submission_df.to_csv("submission.csv", index=False)
```

---

### **Step 7: Load Model & Tokenizer**

Reloaded the trained model and tokenizer for reproducibility and re-inference:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./results")
tokenizer = AutoTokenizer.from_pretrained("./results")
```

---

## 📈 Results

| Metric                     | Score       |
| -------------------------- | ----------- |
| **Public Macro F1-Score**  | 0.49734     |
| **Private Macro F1-Score** | **0.50185** |

Achieved stable performance despite heavy class imbalance by employing:

* Dynamic weighted loss
* Smoothed resampling
* Controlled max sequence lengths

---

## 📚 Key Learnings

* Managing **imbalanced multilingual text classification** efficiently
* Fine-tuning transformer-based LLMs under Kaggle’s runtime constraints
* Incorporating **custom weighted loss functions** into Hugging Face’s Trainer
* Importance of **validation strategy** and **hyperparameter optimization**

---

## 🛠️ Tech Stack

| Category      | Tools / Libraries         |
| ------------- | ------------------------- |
| Language      | Python 3                  |
| Framework     | Hugging Face Transformers |
| Data Handling | Pandas, NumPy, Datasets   |
| Model         | XLM-RoBERTa Base          |
| Visualization | Seaborn, Matplotlib       |
| Metrics       | scikit-learn (F1 Score)   |
| Platform      | Kaggle Notebook           |

---

## 🔍 References

* [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
* [Kaggle Competition Page](https://www.kaggle.com/t/4673999fe3e84d14a17956a4d0c96432)
* [XLM-RoBERTa Paper (Conneau et al., 2020)](https://arxiv.org/abs/1911.02116)

---

## 🧑‍💻 Author

**Parth Bansal**  
BS, IIT Madras | Data Science & Applications  
*“Bridging the past with the intelligence of today.”*  
