import os
os.environ['WANDB_DISABLED'] = 'true'

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from src.data import prepare_datasets
from src.model import encode_labels, tokenize_data
from src.train import ReviewsDataset

from src.evaluate import evaluate_model

HF_REPO = 'sandesh2233/goodreads-genre-distilbert'

train_texts, train_labels, test_texts, test_labels = prepare_datasets()
tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
train_enc, test_enc, label2id, id2label = encode_labels(train_labels, test_labels)
_, test_encodings = tokenize_data(tokenizer, train_texts, test_texts)
test_dataset = ReviewsDataset(test_encodings, test_enc)
model = AutoModelForSequenceClassification.from_pretrained(HF_REPO)
eval_args = TrainingArguments(output_dir='results', per_device_eval_batch_size=8, report_to=[])
trainer = Trainer(model=model, args=eval_args)
evaluate_model(trainer, test_dataset, id2label)
print('Hub evaluation done!')