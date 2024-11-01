from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_path = "./bert_sentiment_model"
tokenizer_path = "./bert_tokenizer"

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_sentiment(text):
    # Preprocessing
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    # Map prediction to sentiment label
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[prediction]

if __name__ == "__main__":
    # Get user input
    user_input = input("Enter text for sentiment analysis: ")
    sentiment = predict_sentiment(user_input)
    print(f"Predicted sentiment: {sentiment}")
