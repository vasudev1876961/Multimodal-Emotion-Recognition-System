import joblib
from src.text.preprocessing import clean_text


def load_models():
    model = joblib.load("models/text_model/model.pkl")
    vectorizer = joblib.load("models/text_model/vectorizer.pkl")
    label_encoder = joblib.load("models/text_model/label_encoder.pkl")

    return model, vectorizer, label_encoder


def predict_emotion(text):
    model, vectorizer, label_encoder = load_models()

    # Clean input
    text = clean_text(text)

    # Convert to vector
    text_vector = vectorizer.transform([text])

    # Predict
    pred = model.predict(text_vector)

    # Convert label back
    emotion = label_encoder.inverse_transform(pred)[0]

    return emotion


if __name__ == "__main__":
    while True:
        user_input = input("Enter text (or 'exit'): ")

        if user_input.lower() == "exit":
            break

        result = predict_emotion(user_input)
        print(f"Predicted Emotion: {result}")