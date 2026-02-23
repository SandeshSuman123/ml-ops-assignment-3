import json
import os
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

os.environ["WANDB_DISABLED"] = "true"


# ================= DATASET CLASS =================

class ReviewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ================= METRICS =================

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}


# ================= TRAINING =================

def train_model(model, train_encodings, test_encodings, train_labels, test_labels):

    train_dataset = ReviewsDataset(train_encodings, train_labels)
    test_dataset  = ReviewsDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        num_train_epochs=1,              
        per_device_train_batch_size=4,  
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        output_dir="./results",
        logging_dir="./logs",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",           
        load_best_model_at_end=True,     
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save model + results
    os.makedirs("saved_model", exist_ok=True)
    trainer.save_model("saved_model")

    results = trainer.evaluate()
    print("\nEvaluation Results:", results)

    #  Save results to JSON so they persist after container exits
    os.makedirs("results", exist_ok=True)
    with open("results/local_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/local_eval_results.json")

    return trainer, results