import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torchtext
import torchtext.data
from torchtext.data.utils import get_tokenizer
import torchtext.datasets
from torchtext.vocab import GloVe
import nltk
import random
from nltk.corpus import wordnet, stopwords
from functools import lru_cache

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="backend/static", html=True), name="static")

# Define the request and response models
class TextRequest(BaseModel):
    dataset: str
    text: str

class TextResponse(BaseModel):
    original_text: str
    processed_text: str

class DatasetRequest(BaseModel):
    dataset: str

# Load datasets and tokenizer
datasets = {
    "AG_NEWS": torchtext.datasets.AG_NEWS,
    "IMDB": torchtext.datasets.IMDB,
    "CoLA": torchtext.datasets.CoLA,
    # Add more datasets as needed
}

tokenizer = get_tokenizer("basic_english")
glove = GloVe(name="6B", dim=100)
stop_words = set(stopwords.words('english'))

logging.basicConfig(level=logging.INFO)

@app.get("/datasets")
def get_datasets():
    logging.info("Fetching datasets")
    return {"datasets": list(datasets.keys())}

@app.post("/fetch_sample")
def fetch_sample(request: DatasetRequest):
    logging.info(f"Fetching sample for dataset: {request.dataset}")
    if request.dataset not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    sample_set = samples_from_ds(request.dataset)
    return {"text": sample_set[random.randint(0, 99)][1]}

@lru_cache(maxsize=10, typed=False)
def samples_from_ds(dataset: str):
    dataset_iter = iter(datasets[dataset](split='train'))
    sample_set = []
    for _ in range(100):
        sample_set.append(next(dataset_iter))
    return sample_set

@app.post("/preprocess")
def preprocess_text(request: TextRequest):
    logging.info(f"Preprocessing text with option: {request.dataset}")
    tokens = tokenizer(request.text)
    if request.dataset == "tokenize":
        processed_text = " ".join([f"[{token}]" for token in tokens])
    elif request.dataset == "pad":
        max_length = 300  # Example padding length
        padded_tokens = tokens + ["<pad>"] * (max_length - len(tokens))
        processed_text = " ".join(padded_tokens[:max_length])
    elif request.dataset == "embed":
        embedded_tokens = [glove[token].tolist() for token in tokens]
        processed_text = str(embedded_tokens)
    else:
        raise HTTPException(status_code=400, detail="Invalid preprocessing option")
    return TextResponse(original_text=request.text, processed_text=processed_text)

@app.post("/augment")
def augment_text(request: TextRequest):
    logging.info(f"Augmenting text with option: {request.dataset}")
    if request.dataset == "synonym_replacement":
        processed_text = synonym_replacement(request.text)
    elif request.dataset == "random_insertion":
        processed_text = random_insertion(request.text)
    elif request.dataset == "random_deletion":
        processed_text = random_deletion(request.text)
    else:
        raise HTTPException(status_code=400, detail="Invalid augmentation option")
    return TextResponse(original_text=request.text, processed_text=processed_text)

@app.get("/", response_class=HTMLResponse)
def get_index():
    with open("backend/static/index.html") as f:
        return f.read()

def synonym_replacement(text):
    words = text.split()
    new_words = []
    for word in words:
        if word.lower() in stop_words or not wordnet.synsets(word):
            new_words.append(word)
            continue
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            if synonym != word:
                new_words.append(synonym)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return " ".join(new_words)

def random_insertion(text, n=1):
    words = text.split()
    for _ in range(n):
        add_word(words)
    return " ".join(words)

def add_word(words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = words[random.randint(0, len(words)-1)]
        synonyms = wordnet.synsets(random_word)
        counter += 1
        if counter > 10:
            return
    synonym = synonyms[0].lemmas()[0].name()
    random_idx = random.randint(0, len(words)-1)
    words.insert(random_idx, synonym)

def random_deletion(text, p=0.2):
    words = text.split()
    if len(words) == 1:
        return words
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if len(new_words) == 0:
        return words[random.randint(0, len(words)-1)]
    return " ".join(new_words)