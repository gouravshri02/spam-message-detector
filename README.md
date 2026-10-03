# 📩 Spam Message Detector

A simple **Machine Learning + Streamlit** college project that detects whether a text message is **Spam** or **Ham (Not Spam)**.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit

## Project Structure

```text
spam_detector_ml_project/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── run.bat
├── data/
│   └── spam.csv
│
└── model files (created after training)
    ├── spam_model.pkl
    └── tfidf_vectorizer.pkl
```

## How to Run on Windows

### 1. Install Python

Use Python 3.10 or 3.11 for a simple setup.

Check Python:

```bash
python --version
```

### 2. Open the project folder

Open Command Prompt/PowerShell inside this folder.

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate it

```bash
venv\Scripts\activate
```

### 5. Install libraries

```bash
pip install -r requirements.txt
```

### 6. Train the ML model

```bash
python train_model.py
```

This creates:

- `spam_model.pkl`
- `tfidf_vectorizer.pkl`

### 7. Start Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

## How the Project Works

```text
User Message
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Prediction
     ↓
Spam / Ham
```

## Example

Input:

```text
Congratulations! You have won a free prize. Call now!
```

Output:

```text
SPAM MESSAGE
```

Input:

```text
Please send me the class notes.
```

Output:

```text
HAM (NOT SPAM)
```

## ML Concepts Used

### 1. TF-IDF

TF-IDF converts text into numerical values that a machine learning algorithm can understand.

### 2. Logistic Regression

Logistic Regression learns patterns from spam and normal messages and predicts the class of a new message.

### 3. Train-Test Split

The dataset is divided into training and testing data. The model learns from the training data and is evaluated using the test data.

## Important Note

This is an educational/minor-project dataset. Real-world spam detection systems normally use much larger datasets and additional preprocessing.
