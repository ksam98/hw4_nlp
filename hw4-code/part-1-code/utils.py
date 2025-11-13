import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import nltk
from nltk.corpus import wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Download necessary NLTK resources if missing 
for resource in ["tokenizers/punkt", "sentiment/vader_lexicon"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        # quiet=True keeps logs clean in training
        if "punkt" in resource:
            nltk.download("punkt", quiet=True)
        else:
            nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()
lexicon = sia.lexicon
detok = TreebankWordDetokenizer()

random.seed(0)

QWERTY_NEIGHBORS = {
    # letters
    'a': list("qwsz"), 'b': list("vghn"), 'c': list("xdfv"),
    'd': list("ersfcx"), 'e': list("wsdr"), 'f': list("rtgdvc"),
    'g': list("tyfhvb"), 'h': list("yugjnb"), 'i': list("ujko"),
    'j': list("uikhmn"), 'k': list("ijolm,"), 'l': list("kop;"),
    'm': list("njk,"), 'n': list("bhjm"), 'o': list("iklp"),
    'p': list("ol;["), 'q': list("wa1"), 'r': list("edft"),
    's': list("awedxz"), 't': list("rfgy"), 'u': list("yhji"),
    'v': list("cfgb"), 'w': list("qase2"), 'x': list("zsdc"),
    'y': list("tghu"), 'z': list("asx"),

    # numbers (include nearby digits and top-row symbols)
    '1': list("2q`"), '2': list("13wq"), '3': list("24we"),
    '4': list("35er"), '5': list("46rt"), '6': list("57ty"),
    '7': list("68yu"), '8': list("79ui"), '9': list("80io"),
    '0': list("9po-"),

    # punctuation and symbols
    '`': list("1q"), '-': list("0p="), '=': list("-p["),
    '[': list("p];"), ']': list("[;'"), ';': list("pl['"),
    "'": list(";]"), ',': list("klm."), '.': list(",/;l"),
    '/': list(".;l"), '\\': list("]"),
    '!': list("@1"), '@': list("!23"), '#': list("@24"),
    '$': list("#35"), '%': list("$46"), '^': list("%57"),
    '&': list("^68"), '*': list("&79"), '(': list("*80"), ')': list("9("),
}

def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    tokens = word_tokenize(text)

    # Find token with highest absolute sentiment score (ie VADER intensity
    token_idx = None
    max_abs_vader_score = 0.0
    for i, token in enumerate(tokens): 
        vader_score = lexicon.get(token.lower(), 0.0)
        if abs(vader_score) > max_abs_vader_score:
            max_abs_vader_score = abs(vader_score)
            token_idx = i

    # Mutate chosen sentiment word with typo
    if token_idx is not None and max_abs_vader_score > 0.0:
        token = tokens[token_idx]
        candidate_idxs = [j for j, ch in enumerate(token) if QWERTY_NEIGHBORS.get(ch.lower())]
        
        if candidate_idxs:
            char_idx = random.choice(candidate_idxs)
            char = token[char_idx]
            new_char = random.choice(QWERTY_NEIGHBORS[char.lower()])
            if char.isupper():
                new_char = new_char.upper()
            typo_token = token[:char_idx] + new_char + token[char_idx+1:]
            tokens[token_idx] = typo_token

    # Reconstruct text
    example["text"] = detok.detokenize(tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
