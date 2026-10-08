# sentiment-analysis-streamlit

# 🎬 IMDB Movie Review Sentiment Analysis

An NLP and Deep Learning project that uses a **Recurrent Neural Network (RNN)** to classify IMDB movie reviews as **Positive** or **Negative**.

The trained model achieved approximately **84% accuracy** on the IMDB dataset and was deployed as an interactive **Streamlit web application**.

## 🚀 Live Demo

👉 Add your Streamlit app link here

## 📌 Features

* 🎬 Classifies movie reviews as **Positive** or **Negative**
* 🧠 Uses a **Recurrent Neural Network (RNN)**
* 📝 Performs text preprocessing
* 🔤 Converts text into numerical sequences
* 📊 Uses the IMDB Movie Review dataset
* 🌐 Interactive Streamlit web interface
* ⚡ Real-time sentiment prediction

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **Pandas**
* **NumPy**
* **NLTK**
* **Scikit-learn**
* **Streamlit**
* **Matplotlib**

## 🧠 Model Architecture

The project uses an RNN-based deep learning architecture for sentiment classification.

```text
Movie Review
     ↓
Text Preprocessing
     ↓
Tokenization
     ↓
Numerical Sequence
     ↓
Embedding Layer
     ↓
RNN
     ↓
Fully Connected Layer
     ↓
Sigmoid
     ↓
Positive / Negative
```

## 📊 Dataset

The project uses the **IMDB Movie Review Dataset**, which contains movie reviews labeled as:

* `Positive`
* `Negative`

The dataset is commonly used for Natural Language Processing and sentiment classification tasks.

## 🔄 Data Preprocessing

The following preprocessing steps are performed:

1. Text cleaning
2. Tokenization
3. Vocabulary creation
4. Conversion of words into numerical IDs
5. Padding/truncating sequences
6. Preparing data for the RNN model

## 🏗️ RNN Model

The neural network processes the review sequence and learns contextual information from the text.

A simplified representation:

```text
Input Text
    ↓
Embedding
    ↓
RNN Layer
    ↓
Hidden Representation
    ↓
Fully Connected Layer
    ↓
Sentiment Prediction
```

## 📈 Model Performance

The model achieved approximately:

**Accuracy: ~84%**

The model was trained on the IMDB dataset and evaluated on unseen review data.

## 🌐 Streamlit Web Application

The trained model was integrated with Streamlit to create a simple web application.

Users can enter a movie review and receive a sentiment prediction.

### Example

```text
Input:
"This movie was absolutely amazing and I loved every scene."

Output:
😊 Positive
```

```text
Input:
"The movie was boring and disappointing."

Output:
😞 Negative
```

## 📂 Project Structure

```text
Sentiment-analysis/
│
├── app.py
├── Rnn_project_sentiment.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

> File names may vary depending on the project version.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Sentiment-analysis
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## ▶️ Run the Streamlit Application

```powershell
streamlit run app.py
```

The application will open in your browser.

## 🎯 Learning Outcomes

Through this project, I worked with:

* Natural Language Processing
* Text preprocessing
* Tokenization
* Word embeddings
* Sequence modeling
* Recurrent Neural Networks
* PyTorch
* Model training and evaluation
* Sentiment classification
* Streamlit deployment

## 🔮 Future Improvements

* Improve model accuracy
* Experiment with **LSTM and GRU**
* Add attention mechanisms
* Compare RNN, LSTM and GRU performance
* Add multilingual sentiment analysis
* Improve UI/UX
* Deploy an optimized production model

## 👨‍💻 Author

**Karan Thakur**

GitHub: https://github.com/karansinghpgl

---

⭐ If you find this project useful, consider giving the repository a star!
