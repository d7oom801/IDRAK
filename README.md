# 🏦 IDRAK | Intelligent Banking Query Router



> **IDRAK (إدراك)** is an end-to-end NLP-based customer intent classification and automated query routing system designed for retail banking support operations.

---

## 📌 Problem Statement
Retail banks receive thousands of unstructured customer support inquiries every day across mobile apps and web chat. Manually reading, tagging, and routing messages creates significant queue delays and increases operational costs per ticket. 

**Our Goal:** Automatically classify single-turn customer messages into **77 fine-grained banking intent categories** in real time, enabling instant automated self-service or direct routing to the appropriate specialist team.

---

## 📊 Dataset: Banking77 (PolyAI)
We utilize the standardized **Banking77** benchmark dataset consisting of **13,083** real-world customer inquiries:
* **Training Set:** 10,003 queries (~130 examples per intent class).
* **Test Set:** 3,080 queries (strictly stratified with exactly 40 test queries per class).
* **Target ($y$):** 77 fine-grained intent categories (e.g., `lost_or_stolen_card`, `pending_transfer`, `card_arrival`, `fee_inquiry`).
* **Data Quality:** Zero missing values, zero duplicate entries, and zero data leakage between train and test splits.
* **Message Length:** Average of **11.9 words** per customer query (median: 10 words, max: 79 words).

---

## 🧠 Modeling & Machine Learning Approach
We built a leak-free Scikit-Learn `Pipeline` fitted strictly on training data:
1. **Feature Extraction:** `TfidfVectorizer` capturing both unigrams and bigrams (`ngram_range=(1, 2)`), top 10,000 features, and sublinear term frequency scaling (`sublinear_tf=True`).
2. **Classifier:** Multinomial `LogisticRegression` ($C=5.0$, `max_iter=1000`, `class_weight='balanced'`).
3. **Cross-Validation:** 5-Fold Stratified Cross-Validation on training data to ensure model stability and prevent overfitting.

---

## 📈 Key Results & Evaluation

| Metric | Score (%) | Context / Benchmark |
| :--- | :---: | :--- |
| **Dummy Baseline** | **1.87%** | Most frequent class prediction (random baseline floor) |
| **5-Fold CV Accuracy** | **87.58% (±0.61%)** | Highly stable across training folds (zero overfitting) |
| **Top-1 Test Accuracy** | **89.42%** | Correct intent predicted on the first attempt on unseen test queries |
| **Top-3 Test Accuracy** | **97.08%** | Correct intent is within the top 3 recommendations |
| **Macro F1-Score** | **89.46%** | Balanced precision and recall across all 77 intent categories |

### 🔍 Error Analysis (Semantic Overlap)
A detailed error analysis revealed that over 60% of misclassifications occur between intent pairs sharing near-identical vocabulary and subtle temporal nuances (e.g., `card_arrival` vs. `card_delivery_estimate`). In production banking environments, these errors carry minimal operational risk because both intents route to the exact same department (Card Logistics).

---

## 📁 Repository Structure

```text
.
├── IDRAK.ipynb               # Full end-to-end pipeline (Day 1: EDA, Day 2: Baseline, Day 3: Diagnostics)
├── demo.ipynb                # Lightweight demo notebook with single-cell human-readable prediction
├── Dashboard_Demo.py         # Streamlit interactive query routing web application
├── banking_pipeline.joblib   # Serialized Scikit-Learn model pipeline
├── categories.joblib         # Serialized list of 77 intent category labels
├── train_clean.csv           # Clean training dataset split (10,003 rows)
├── test_clean.csv            # Clean held-out test dataset split (3,080 rows)
├── word_length_distribution.png  # Message length distribution chart
├── top_confused_intents.png      # Top confused intent pairs chart
└── README.md                 # Project documentation

