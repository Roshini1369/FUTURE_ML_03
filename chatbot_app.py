import os
import zipfile
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Step 1: Extract CSV from ZIP if not already
if not os.path.exists("customer_support_tickets.csv"):
    with zipfile.ZipFile("customer_support_tickets.csv.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# Step 2: Load and preprocess data
file_path = "customer_support_tickets.csv"
df = pd.read_csv(file_path)
df = df.dropna(subset=["Ticket Description", "Ticket Type", "Resolution"])
texts = df["Ticket Description"].values
labels = df["Ticket Type"].values
responses = df["Resolution"].values

le = LabelEncoder()
y = le.fit_transform(labels)
y_cat = to_categorical(y)

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts).toarray()

# Step 3: Build or load model
MODEL_PATH = "chatbot_model.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ Model loaded from disk.")
else:
    print("🧠 Training model...")
    model = Sequential()
    model.add(Dense(128, input_shape=(X.shape[1],), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(y_cat.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y_cat, epochs=20, batch_size=8, verbose=1)
    model.save(MODEL_PATH)
    print("💾 Model saved.")

# Step 4: Inference logic
def predict_intent(text):
    vec = vectorizer.transform([text]).toarray()
    pred = model.predict(vec, verbose=0)[0]
    intent = le.inverse_transform([np.argmax(pred)])[0]
    return intent

intent_to_response = df.groupby("Ticket Type")["Resolution"].apply(lambda x: x.sample(1).values[0]).to_dict()

def get_bot_response(msg):
    intent = predict_intent(msg)
    return intent_to_response.get(intent, "Sorry, I didn’t get that. Can you rephrase?")

# Step 5: Flask web app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_msg = request.form["msg"]
    bot_reply = get_bot_response(user_msg)
    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
