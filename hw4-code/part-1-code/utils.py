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

# QWERTY keyboard neighbors for simulating typos
# NOTE: this was generated with the help of LLMs (ChatGPT 5.1) with the following
# prompt: "For a QWERTY keyboard, provide a dictionary where each key is a letter, number,
# punctuation or symbol, and the value is a list of nearby keys on the keyboard that could
# be used to simulate a typo. Provide the dictionary as valid Python code."
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

def _apply_typo(word: str) -> str:
    """Pick a position that has QWERTY neighbors and substitute one char (no fallback)."""
    idxs = [i for i, ch in enumerate(word) if QWERTY_NEIGHBORS.get(ch.lower())]
    if not idxs:
        return word  
    j = random.choice(idxs)
    ch = word[j]
    repl = random.choice(QWERTY_NEIGHBORS[ch.lower()])
    if ch.isupper():
        repl = repl.upper()
    return word[:j] + repl + word[j+1:]

# NOTE: The following function was primarily generated with the help of LLMs (ChatGPT 5.1)
# with minor edits from me. The following prompt was used: 
# "Create a function _synonym_swap that takes a word as input and replaces it with a synonym
# from WordNet. The function should preserve the case style of the original word (e.g
# uppercase, capitalized, lowercase)."
def _synonym_swap(word: str) -> str:
    """
    Replace with a WordNet synonym, including multi-word lemmas.
    Underscores are converted into spaces.
    Preserve case style.
    """
    base = word.lower()
    synsets = wordnet.synsets(base)
    repl = None

    for synset in synsets:

        if not synset:
            continue

        for lemma in synset.lemmas():
            name = lemma.name()  

            # Ignore exact identical lemma
            if name.lower() == base:
                continue
            
            # Allow alphabetic + underscores (multi-word synonyms)
            if not all(ch.isalpha() or ch == "_" for ch in name):
                continue

            repl = name.replace("_", " ")  # convert to natural spacing
            break

        if repl:
            break

    if not repl:
        return word

    # Preserve original word's casing
    if word.isupper():
        repl = repl.upper()
    elif word[0].isupper():
        repl = repl.capitalize()
    else:
        repl = repl.lower()

    return repl

def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    tokens = word_tokenize(text)

    # NOTE: I used the following prompt with ChatGPT 5.1 "I am working on machine learning 
    # problem to classify different review to sentiment. To aid data augmentation and processing,
    # give me strategies to identify sentiment words in a sentence. Ideally use python's NLTK, 
    # pytorch, hugging face libraries etc to do so." to learn about VADER and how to use it.

    # Rank and sort (descending) tokens by absolute VADER intensity
    idx_vader_score_tuples = []
    for i, token in enumerate(tokens):
        vader_intensity = lexicon.get(token.lower(), 0.0)
        if vader_intensity != 0.0:
            idx_vader_score_tuples.append((i, abs(vader_intensity)))

    if not idx_vader_score_tuples:
        # no sentiment-bearing tokens; leave unchanged
        example["text"] = detok.detokenize(tokens)
        return example

    idx_vader_score_tuples.sort(key=lambda x: x[1], reverse=True)
    indices = [idx for idx, _ in idx_vader_score_tuples]
    
    if indices:
        # tune K depending on how aggressive you want to be
        max_strong = min(10, len(indices))
        for idx in indices[:max_strong]:
            w = _synonym_swap(tokens[idx])  # may return original if no synonym
            n_typos = random.choice([1, 2])
            for _ in range(n_typos):
                w = _apply_typo(w)
            tokens[idx] = w

    # Reconstruct text
    example["text"] = detok.detokenize(tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
