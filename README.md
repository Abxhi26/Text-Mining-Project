# 📊 Text Mining Projects Repository

## 📌 Overview

This repository contains multiple **Text Mining and Natural Language Processing (NLP)** projects developed as part of an academic lab.
Each project focuses on different applications of text mining such as **classification, clustering, and detection systems**.

These projects demonstrate how unstructured textual data can be processed, analyzed, and used to build intelligent systems.

---

## 🧠 Technologies Used

* Python
* Scikit-learn
* NLTK
* Pandas
* TF-IDF Vectorization
* Machine Learning Algorithms (Logistic Regression, Naive Bayes, K-Means)

---

# 📁 Projects Included

---

## 🔹 1. Spam Classification with Explainability

**👨‍💻 Author:** Abhiram Aravind 22MIC0170

### 📌 Description

This project focuses on classifying SMS messages as **Spam 🚨 or Not Spam ✅** using machine learning techniques.

It also includes an **explainability module** that highlights important words influencing predictions, making the model transparent and interpretable.

### ⚙️ Methodology

* Text preprocessing (cleaning, stopword removal)
* TF-IDF feature extraction
* Logistic Regression / Naive Bayes classification
* Feature importance analysis for explainability

### 📊 Dataset

* SMS Spam Collection Dataset
* Contains labeled messages (spam/ham)

### 🚀 Features

* Interactive input system
* Real-time classification
* Confidence score
* Important word explanation

### 📈 Applications

* SMS spam filtering
* Email filtering systems
* Fraud and phishing detection

---

## 🔹 2. Text Clustering for Document Analysis

**👩‍💻 Author:** Manaswitha G 22MIC0127

### 📌 Description

This project applies **unsupervised learning (clustering)** to group similar text documents without labels.

It uses TF-IDF and K-Means to identify patterns in text data such as sentiment grouping and topic detection.

### ⚙️ Methodology

* Text preprocessing
* TF-IDF vectorization
* K-Means clustering
* Elbow Method (optimal clusters)
* Silhouette Score evaluation

### 📊 Dataset

* IMDB Movie Reviews Dataset

### 📈 Output

* Clustered documents
* Keyword extraction
* WordCloud visualizations

👉 According to the analysis, clustering effectively groups reviews based on sentiment and vocabulary patterns 

---

## 🔹 3. Fake Review Detection System

**👩‍💻 Author:** Riya Khandelwal 22MID0253

### 📌 Description

This project detects whether a review is **Fake (Deceptive)** or **Genuine (Truthful)** using machine learning.

It addresses the growing problem of fake reviews in e-commerce platforms.

### ⚙️ Methodology

* Text preprocessing
* Bag-of-Words / TF-IDF feature extraction
* Naive Bayes classification
* Model evaluation using multiple metrics

### 📊 Dataset

* Deceptive Opinion Spam Corpus
* 800 labeled hotel reviews

### 📈 Performance

* Achieves ~88% accuracy
* Outperforms human detection (~57%) 

### 📈 Applications

* E-commerce platforms
* Review moderation systems
* Fraud detection

---

# 🔄 Common Workflow Across Projects

1. Data Collection
2. Text Preprocessing
3. Feature Extraction (TF-IDF / BoW)
4. Model Training / Clustering
5. Evaluation
6. Interpretation / Visualization

---

# 🎯 Key Learnings

* Text mining enables extraction of meaningful insights from unstructured data
* TF-IDF is a powerful baseline feature extraction technique
* Machine learning models can effectively classify and group text
* Explainability improves trust in AI systems

---

# ⚠️ Limitations

* Performance depends on dataset quality
* TF-IDF does not capture deep semantic meaning
* Models may struggle with unseen or modern patterns

---

# 🔮 Future Scope

* Use advanced models (BERT, LSTM)
* Build web-based interfaces (Streamlit)
* Expand datasets for better generalization
* Deploy real-time systems

---

# 👨‍💻 Contributors

* **Abhiram Aravind** – Spam Classification
* **Manaswitha G** – Text Clustering
* **Riya Khandelwal** – Fake Review Detection

---

# 🏁 Conclusion

This repository showcases how **text mining techniques combined with machine learning** can solve real-world problems such as spam detection, document clustering, and fake review identification.

Each project highlights a different aspect of NLP, making this collection a comprehensive demonstration of text mining applications.
