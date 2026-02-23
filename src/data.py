import gzip
import json
import os
import pickle
import random
import requests

# ===================== CONFIG =====================

GENRE_URLS = {
    'poetry':                 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children':               'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic':         'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal':     'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography':      'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance':                'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult':            'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz',
}

CACHE_PATH = "data/reviews_cache.pickle"


# ===================== DOWNLOAD =====================

def load_reviews(url, head=10000, sample_size=2000):
    reviews = []
    count = 0

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()  

    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            text = d.get('review_text', '').strip()  
            if text:
                reviews.append(text)
                count += 1
            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def download_all_genres(force_download=False):
    """Download reviews for all genres, with caching."""
    if not force_download and os.path.exists(CACHE_PATH):
        print(f"Loading cached data from {CACHE_PATH}")
        with open(CACHE_PATH, 'rb') as f:
            return pickle.load(f)

    os.makedirs("data", exist_ok=True)
    genre_reviews = {}

    for genre, url in GENRE_URLS.items():
        print(f"Downloading: {genre}...")
        genre_reviews[genre] = load_reviews(url)

    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(genre_reviews, f)
    print(f"Data cached to {CACHE_PATH}")

    return genre_reviews


# ===================== SPLIT =====================

def prepare_datasets(per_genre=120, train_size=100):
    genre_reviews = download_all_genres()

    train_texts, train_labels = [], []
    test_texts,  test_labels  = [], []

    for genre, reviews in genre_reviews.items():
        subset = random.sample(reviews, min(per_genre, len(reviews)))

        for r in subset[:train_size]:
            train_texts.append(r)
            train_labels.append(genre)

        for r in subset[train_size:]:
            test_texts.append(r)
            test_labels.append(genre)

    print(f"Train: {len(train_texts)} reviews | Test: {len(test_texts)} reviews")
    return train_texts, train_labels, test_texts, test_labels  


if __name__ == "__main__":
    train_texts, train_labels, test_texts, test_labels = prepare_datasets()
    print("Sample:", train_labels[0], "→", train_texts[0][:80])
