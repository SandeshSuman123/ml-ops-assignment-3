# Goodreads Genre Classifier — DistilBERT

Fine-tuned DistilBERT model for classifying Goodreads book reviews into 8 genres using HuggingFace Trainer API, containerized with Docker.

## Genres
`poetry` · `children` · `comics_graphic` · `fantasy_paranormal` · `history_biography` · `mystery_thriller_crime` · `romance` · `young_adult`

## Model Selection
**DistilBERT** (`distilbert-base-uncased`) was chosen because:
- 40% smaller and 60% faster than full BERT
- Ideal for CPU training
- Retains 97% of BERT's performance

## Project Structure
```
├── Dockerfile
├── Dockerfile.eval
├── requirements.txt
├── main.py
├── hub_eval.py
└── src/
    ├── data.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    └── utils.py
```

## Docker Build Instructions
```bash
docker build -t hf-classifier .
docker run -v ${PWD}/results:/app/results -v ${PWD}/saved_model:/app/saved_model hf-classifier
```

## Evaluation Results
| Metric | Local Model | Hub Model |
|--------|-------------|-----------|
| Accuracy | 0.50 | 0.12 |
| Eval Loss | 1.64 | N/A |

## HuggingFace Model
[sandesh2233/goodreads-genre-distilbert](https://huggingface.co/sandesh2233/goodreads-genre-distilbert)

## Challenges
- CPU training was slow, limited to 1 epoch
- Label mapping inconsistency between local and hub evaluation
- Docker disk space management on Windows
