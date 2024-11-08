import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torchtext
from torchtext.data.utils import get_tokenizer
import torchtext.datasets
from torchtext.vocab import GloVe
import nltk
import random
from nltk.corpus import wordnet

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
    # Add more datasets as needed
}

tokenizer = get_tokenizer("basic_english")
glove = GloVe(name="6B", dim=100)

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
    dataset_iter = datasets[request.dataset](split='train')
    sample = next(iter(dataset_iter))
    return {"text": sample[1]}

@app.post("/preprocess")
def preprocess_text(request: TextRequest):
    logging.info(f"Preprocessing text with option: {request.dataset}")
    tokens = tokenizer(request.text)
    if request.dataset == "tokenize":
        processed_text = " ".join(tokens)
    elif request.dataset == "pad":
        max_length = 1000  # Example padding length
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
    else:
        raise HTTPException(status_code=400, detail="Invalid augmentation option")
    return TextResponse(original_text=request.text, processed_text=processed_text)

@app.get("/", response_class=HTMLResponse)
def get_index():
    with open("backend/static/index.html") as f:
        return f.read()

def synonym_replacement(text):
    words = text.split()
    for _ in range(10):
        word_to_replace = random.choice(words)
        synonyms = wordnet.synsets(word_to_replace)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            words = [synonym if word == word_to_replace else word for word in words]
    return " ".join(words)

def random_insertion(text):
    # Dummy implementation for random insertion
    words = text.split()
    words.insert(len(words) // 2, "random")
    return " ".join(words)