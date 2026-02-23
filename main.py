from src.evaluate import evaluate_model
from src.data import prepare_datasets
from src.model import load_tokenizer, encode_labels, tokenize_data, load_model
from src.train import train_model


def main():
    print("Preparing dataset...")
    train_texts, train_labels, test_texts, test_labels = prepare_datasets()

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Encoding labels...")
    train_labels_enc, test_labels_enc, label2id, id2label = encode_labels(train_labels, test_labels)

    print("Tokenizing text...")
    train_encodings, test_encodings = tokenize_data(tokenizer, train_texts, test_texts)

    print("Loading model...")
    model = load_model(len(label2id), id2label, label2id)

    print("Training model...")
    trainer, results = train_model(
        model, train_encodings, test_encodings,
        train_labels_enc, test_labels_enc
    )

    print("Evaluating model...")
    evaluate_model(trainer, trainer.eval_dataset, id2label)


if __name__ == "__main__":
    main()