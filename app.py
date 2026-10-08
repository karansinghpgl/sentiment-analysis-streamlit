import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# -----------------------------
# NLTK resources
# -----------------------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# -----------------------------
# RNN MODEL
# -----------------------------
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size
        )

        out, _ = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])

        return out


# -----------------------------
# LOAD TF-IDF
# -----------------------------
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)


# -----------------------------
# CREATE MODEL
# -----------------------------
input_size = len(tfidf.get_feature_names_out())

model = RNN(
    input_size=input_size,
    hidden_size=128,
    num_layers=1
)


# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()


# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopwords
    stop_words = stopwords.words("english")

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Stemming
    ps = PorterStemmer()

    tokens = [
        ps.stem(word)
        for word in tokens
    ]

    return " ".join(tokens)


# -----------------------------
# STREAMLIT PAGE
# -----------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)


st.title("😊 Sentiment Analysis")

st.write(
    "Enter a review below and our RNN model "
    "will predict its sentiment."
)


# -----------------------------
# INPUT
# -----------------------------
review = st.text_area(
    "Enter your review:",
    placeholder="Example: This movie was absolutely amazing!",
    height=150
)


# -----------------------------
# PREDICT
# -----------------------------
if st.button("🔍 Predict Sentiment"):

    if not review.strip():

        st.warning("Please enter a review.")

    else:

        # Preprocess
        cleaned_text = preprocess_text(review)

        # TF-IDF transformation
        vector = tfidf.transform(
            [cleaned_text]
        ).toarray()

        # Convert to tensor
        X = torch.tensor(
            vector,
            dtype=torch.float32
        )

        # RNN expects:
        # batch_size × sequence_length × input_size

        X = X.unsqueeze(1)


        # Prediction
        with torch.no_grad():

            output = model(X)

            probability = torch.sigmoid(
                output.squeeze()
            ).item()


        # -----------------------------
        # RESULT
        # -----------------------------

        if probability >= 0.5:

            st.success("😊 POSITIVE SENTIMENT")

            confidence = probability * 100

        else:

            st.error("😞 NEGATIVE SENTIMENT")

            confidence = (1 - probability) * 100


        st.progress(
            int(confidence)
        )

        st.write(
            f"**Confidence: {confidence:.2f}%**"
        )