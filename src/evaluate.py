import os
from sklearn.metrics import classification_report, accuracy_score


def evaluate_model(trainer, test_dataset, id2label):
    print("\nRunning evaluation...")

    predictions = trainer.predict(test_dataset)
    preds  = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    acc = accuracy_score(labels, preds)
    print(f"\nAccuracy: {acc:.4f}")

    # Convert integer IDs back to genre name strings
    preds_named  = [id2label[p] for p in preds]
    labels_named = [id2label[l] for l in labels]

    report = classification_report(labels_named, preds_named)
    print("\nClassification Report:")
    print(report)

    # Save results to results/ folder so Docker volume mount preserves them
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print("Results saved to results/evaluation.txt")

    return {"accuracy": acc}  # ✅ caller can use this value
