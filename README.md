# 📰 Fake News Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-NLP-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

A Machine Learning project that classifies news articles as **Fake** or **Real**
using Natural Language Processing (NLP) techniques.

---

## 🚀 Features
- Text preprocessing and cleaning
- TF-IDF feature extraction
- Machine Learning classification
- Confidence score for predictions
- Streamlit-based interactive UI

---

## 🧠 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Streamlit

---

## 📂 Dataset

Due to GitHub file size limitations, the dataset is not included in this repository.

You can download the dataset from Kaggle:

🔗 Fake and Real News Dataset  
https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

Fake-news-detection-ml/  
│  
├── data/  
│ ├── Fake.csv  
│ └── True.csv  

### Steps after download:
1. Download `Fake.csv` and `True.csv`
2. Create a folder named `data/` in the project root
3. Place the files as shown below:


---

## ⚙️ How to Run the Project  

### 1️⃣ Clone the repository
```
git clone https://github.com/DeerajVanaparthi/fake-news-detection-ml.git
cd fake-news-detection-ml
```
### 2️⃣ Create and activate a virtual environment
```
python -m venv venv
```


Activate it:

Windows
```
venv\Scripts\activate
```

Linux / macOS
```
source venv/bin/activate
```

### 3️⃣ Install project dependencies
```
pip install -r requirements.txt
```

### 5️⃣ Download and place the dataset  

Download the dataset from Kaggle:  
```
https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
```
After extracting, place the files in the following structure:  

fake-news-detection-ml/  
│  
├── data/  
│   ├── Fake.csv  
│   └── True.csv  

▶️ How to Run the Project

### 1️⃣ Train the machine learning model
```
cd app
python train.py
```

This step trains the model and saves:

 - `fake_news_model.pkl`

- `tfidf_vectorizer.pkl`

### 2️⃣ Run the web application
```
streamlit run ui.py
```

#### The application will open automatically in your browser at:

`http://localhost:8501`
Paste a news article into the input box and click Predict to view the result.
