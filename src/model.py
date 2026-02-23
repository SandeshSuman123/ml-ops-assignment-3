import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased" 
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def encode_labels(train_labels, test_labels):
    unique_labels = sorted(set(train_labels))  
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label  = {i: label for label, i in label2id.items()}

    train_encoded = [label2id[y] for y in train_labels]
    test_encoded  = [label2id[y] for y in test_labels]

    return train_encoded, test_encoded, label2id, id2label


def tokenize_data(tokenizer, train_texts, test_texts):
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )
    return train_encodings, test_encodings


def load_model(num_labels, id2label, label2id):  
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,    
        label2id=label2id,
    )
    return model.to(DEVICE)
