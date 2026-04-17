import re
import pickle
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# ===============================
# 📦 LOAD SAVED MODEL
# ===============================
with open('../model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("✅ Model loaded successfully!\n")

# ===============================
# 🧹 PREPROCESSING
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# ===============================
# 🔍 PREDICTION
# ===============================
def predict_message(message):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    result = "SPAM 🚨" if prediction == 1 else "NOT SPAM ✅"
    confidence = max(probabilities)

    # Explainability
    feature_names = vectorizer.get_feature_names_out()
    vector_array = vector.toarray()[0]

    word_scores = []
    for i, score in enumerate(vector_array):
        if score > 0:
            word_scores.append((feature_names[i], score))

    word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)

    return result, confidence, word_scores[:5]

# ===============================
# 🖥️ INTERACTIVE LOOP
# ===============================
while True:
    user_input = input("\nEnter a message (or type 'exit'): ")

    if user_input.lower() == 'exit':
        print("👋 Exiting...")
        break

    result, confidence, explanation = predict_message(user_input)

    print("\n🔍 Prediction:", result)
    print("📊 Confidence:", round(confidence * 100, 2), "%")

    print("\n🧠 Important words:")
    if explanation:
        for word, score in explanation:
            print(f"{word} → score: {round(score, 3)}")
    else:
        print("No strong keywords found.")