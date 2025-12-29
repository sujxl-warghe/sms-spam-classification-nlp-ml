# 💌 Email / SMS Spam Classifier

A Machine Learning–powered **Spam Detection System** that classifies text messages and emails as **Spam** or **Not Spam (Ham)**.  
This project uses **Natural Language Processing (NLP)** techniques and multiple ML models — and is deployed as a beautiful, interactive **Streamlit Web App**.

---

## 🚀 Features

✔️ Classify any text message instantly  
✔️ High-accuracy ML Model trained on real spam datasets  
✔️ Clean NLP pipeline (tokenization, stopwords removal, stemming, TF-IDF)  
✔️ Multiple ML models evaluated & compared  
✔️ Gmail Inbox Integration — fetch and classify your emails  
✔️ Spam Probability Score  
✔️ User-friendly UI with mobile support  
✔️ Built using **Python + Streamlit + Scikit-Learn**

---


## 🧠 Tech Stack

- **Python**
- **Scikit-Learn**
- **Pandas / NumPy**
- **NLTK**
- **Streamlit**
- **IMAP (Gmail API compatible)**
- **BeautifulSoup**
- **XGBoost / Random Forest**
- **Pickle (Model + Vectorizer storage)**

---

## 🧹 NLP Pipeline

Each message is preprocessed using:

✔ Convert to lowercase  
✔ Tokenization  
✔ Remove punctuation  
✔ Remove stop-words  
✔ Keep only alphanumeric words  
✔ Apply stemming (Porter Stemmer)  
✔ Convert to vector using **TF-IDF**

This ensures only meaningful words are passed to the ML model.

---

## 🤖 Machine Learning Models Used

Multiple classifiers were trained & compared:

- Logistic Regression  
- Multinomial Naive Bayes  
- Support Vector Classifier  
- Random Forest  
- Decision Tree  
- KNN  
- Gradient Boosting  
- AdaBoost  
- Bagging Classifier  
- ExtraTrees Classifier  
- XGBoost  

Performance was evaluated using:

📌 **Accuracy**  
📌 **Precision (important for spam filtering)**  

The best performing model was selected for deployment.
## 📊 Model Performance Comparison

Multiple Machine Learning classifiers were trained and evaluated on the spam-classification dataset.  
Both **Accuracy** and **Precision** were used as evaluation metrics. Precision is especially important in spam detection because it measures how many predicted spam messages were actually spam — reducing false spam alerts.

| Algorithm   | Accuracy  | Precision |
|-------------|----------:|----------:|
| KNN         | 0.9062    | **1.0000** |
| Naive Bayes | 0.9565    | **1.0000** |
| SVC         | **0.9720** | 0.9900 |
| Random Forest | 0.9691  | 0.9897 |
| Extra Trees | 0.9691    | 0.9703 |
| Logistic Regression | 0.9536 | 0.9540 |
| XGBoost     | 0.9710    | 0.9533 |
| Gradient Boosting | 0.9565 | 0.9020 |
| Bagging Classifier | 0.9594 | 0.8512 |
| Decision Tree | 0.9410 | 0.8173 |
| AdaBoost    | 0.9246    | 0.7692 |

### 🏆 Best Performing Models
✔ **SVC achieved the highest overall accuracy (~97.2%)**  
✔ **Naive Bayes & KNN achieved perfect precision (1.0)**  
✔ **Random Forest & Extra Trees** also performed strongly


---

## 📊 Model Training Workflow

1️⃣ Load dataset  
2️⃣ Clean & preprocess text  
3️⃣ Convert text → TF-IDF vectors  
4️⃣ Train multiple ML models  
5️⃣ Compare performance  
6️⃣ Save best model & vectorizer using Pickle  
7️⃣ Deploy via Streamlit  

---

## 🌐 Gmail Inbox Spam Detection

The app allows users to:

📥 Login using secure **App Password**  
🔍 Fetch emails from Gmail Inbox  
🤖 Classify each mail as Spam / Not Spam  
📊 View spam probability score  
🧾 Get summary analytics  
👀 Preview last 5 emails  

> 🔐 *Note: Gmail App Password is required — normal password won’t work.*

Help Guide:  
https://support.google.com/accounts/answer/185833

---
```text
spam-classifier/
│
├── app.py
├── sms-spam-model-building.ipynb
├── README.md
└── requirements.txt
```
---

## 📦 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/sujxl-warghe/sms-spam-classification-nlp-ml.git
```
### 2️⃣ Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Download NLTK resources
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```
### 5️⃣ Run the App
```bash
5️⃣ Run the App
```