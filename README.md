# 🚀 Advanced NLP Solutions: Business-Focused Machine Learning Portfolio

Welcome to my Applied Natural Language Processing (NLP) portfolio. This repository contains end-to-end machine learning pipelines designed to solve real-world business challenges, spanning fraud detection, operational triage, and customer experience analytics. 

Each project demonstrates a full data science lifecycle: from raw data extraction and messy text preprocessing to feature engineering, model training, and business-driven evaluation.

---

## 📁 Project Directory

### 1. Customer Grievance Classifier (Financial Services)
* **Goal:** Automate the routing of consumer financial complaints to the correct corporate departments to reduce manual triage overhead.
* **Methodology:** 
  * Cleaned a massive, heavily-redacted dataset using custom Regular Expressions (Regex).
  * Handled severe target class imbalance by consolidating legacy categories and applying `class_weight='balanced'`.
  * Vectorized text using Memory-Optimized TF-IDF.
* **Business Impact:** Built a Logistic Regression classification system that accurately predicts minor, infrequent complaint categories that standard baseline models completely missed, ensuring high-risk grievances are routed correctly.
* **Notebook:** [`Customer_Grievance_Classifier.ipynb`](./Customer_Grievance_Classifier.ipynb)

### 2. Fake Job Posting Detection (Fraud Analytics)
* **Goal:** Identify fraudulent job postings to protect job seekers and minimize platform liability.
* **Methodology:** 
  * Programmatically engineered a unified text feature by concatenating 16 disparate tabular columns (titles, company profiles, descriptions, requirements).
  * Handled imbalanced fraud data and trained tree-based ensemble models (Decision Trees and Random Forests).
* **Business Impact:** Deployed a Random Forest Classifier achieving over 97% accuracy with high weighted-recall, successfully isolating fraudulent posts while minimizing false positives for legitimate employers.
* **Notebook:** [`Fake_Job_Posting_Detection.ipynb`](./Fake_Job_Posting_Detection.ipynb)

### 3. E-Commerce Sentiment Analysis (Amazon Food Reviews)
* **Goal:** Extract actionable customer sentiment from unstructured e-commerce product reviews.
* **Methodology:** 
  * Conducted EDA on a dataset of over 500,000 reviews.
  * Applied NLTK's VADER lexicon for initial sentiment scoring.
* **Business Impact:** Identified that a standard 5-star rating scale was adding noise and causing low model accuracy. Strategically re-engineered the target variable into a 3-tier sentiment system (Positive/Neutral/Negative), boosting final model performance to 90% accuracy.
* **Notebook:** [`Amazon_Food_Reviews_Sentiment.ipynb`](./Amazon_Food_Reviews_Sentiment.ipynb)

---

## 🛠️ Core Technologies & Skills
* **Languages:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, NLTK, VADER, Matplotlib, Seaborn
* **Techniques:** Natural Language Processing (NLP), TF-IDF Vectorization, Regex Text Cleaning, Word Sense Disambiguation, Feature Engineering, Imbalanced Data Handling
* **Deployment:** Pipeline Serialization (`joblib`)

---
