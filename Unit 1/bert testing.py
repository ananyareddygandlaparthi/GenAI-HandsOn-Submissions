
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch

model_path = r""


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

print("🔄 Loading model and tokenizer...")
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.to(device)
model.eval()
print("✅ Model loaded successfully!")

label_encoder = LabelEncoder()
label_encoder.fit(["drug and alcohol", "early life", "personality", "trauma and stress"])
label_names = label_encoder.classes_

def predict_text(text):
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).to(device)

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).item()

    return label_names[preds]

print("\n💬 Mental Health Cause Detector — type 'exit' to stop\n")

while True:
    user_input = input("Enter a sentence: ").strip()
    if user_input.lower() == "exit":
        print("Exiting...")
        break

    label = predict_text(user_input)
    print(f"🔹 Predicted Label: {label}\n")
