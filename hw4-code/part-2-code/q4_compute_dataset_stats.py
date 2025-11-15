# NOTE: this script was primarily generated with the help of LLMs (ChatGPT 5.1). The 
# responses to the following two prompts formed the basis of the script with some minor edits by me:
#
# Prompt 1: [After completing train_t5.py and load_data.py, I uploaded them both to ChatGPT]:
# "I’ve attached train_t5.py and load_data.py, understand the data loading and preprocessing steps 
# in load_data.py [additionally note that there exists a /data folder with dev.nl, dev.sql, train.nl, 
# train.sql files], and the training loop is in train_t5.py. I need you to write a standalone Python 
# script that computes dataset statistics before and after preprocessing. Specifically, I need you to 
# compute the following: number of examples, mean natural language length (token count), mean SQL length, 
# natural language vocab size (unique token IDs) and SQL vocab size"
# 
# Prompt 2: 
# "For computing after preprocessing statistics, Instead of re-implementing the preprocessing logic,
# please use the existing data loading and preprocessing code in load_data.py to create dataloaders
# for the train and dev sets, and then compute the required statistics from the dataloaders"

import os
import numpy as np
import torch
from transformers import T5TokenizerFast
from load_data import load_lines, load_t5_data, PAD_IDX  

DATA_DIR = "data"

def compute_stats_before(split, tokenizer):
    nl_path = os.path.join(DATA_DIR, f"{split}.nl")
    sql_path = os.path.join(DATA_DIR, f"{split}.sql")

    nl_lines = load_lines(nl_path)
    sql_lines = load_lines(sql_path)

    # Tokenize with T5 tokenizer (no truncation here)
    nl_token_lists = [
        tokenizer.encode(x, add_special_tokens=True, truncation=False)
        for x in nl_lines
    ]
    sql_token_lists = [
        tokenizer.encode(x, add_special_tokens=True, truncation=False)
        for x in sql_lines
    ]

    # Example counts
    num_examples = len(nl_lines)

    # Mean lengths
    mean_nl_len = np.mean([len(toks) for toks in nl_token_lists])
    mean_sql_len = np.mean([len(toks) for toks in sql_token_lists])

    # Vocabulary sizes (unique token IDs)
    nl_vocab = set(tok for seq in nl_token_lists for tok in seq)
    sql_vocab = set(tok for seq in sql_token_lists for tok in seq)

    return {
        "num_examples": num_examples,
        "mean_nl_len": float(mean_nl_len),
        "mean_sql_len": float(mean_sql_len),
        "nl_vocab_size": len(nl_vocab),
        "sql_vocab_size": len(sql_vocab),
    }

def collect_stats_after(loader, pad_idx=PAD_IDX):
    """
    Compute AFTER-preprocessing statistics for a T5 dataloader:
      - mean encoder length
      - encoder vocab size
      - mean decoder length (if targets exist)
      - decoder vocab size

    Returns a dict with the 4 fields.
    """

    enc_lengths = []
    enc_vocab = set()

    dec_lengths = []
    dec_vocab = set()

    for batch in loader:
        # Handle both tuple-style and dict-style batches
        if isinstance(batch, dict):
            encoder_ids = batch["encoder_ids"]          # B × T
            encoder_mask = batch["encoder_mask"]        # B × T
            decoder_targets = batch.get("decoder_targets", None)
        else:
            # Assuming collate_fn returns:
            # (encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs)
            # or similar; we only need the first 4 at most.
            if len(batch) == 5:
                encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs = batch
            elif len(batch) == 4:
                encoder_ids, encoder_mask, decoder_inputs, decoder_targets = batch
            elif len(batch) == 2:
                encoder_ids, encoder_mask = batch
                decoder_targets = None
            else:
                raise ValueError(f"Unexpected batch structure with length {len(batch)}: {type(batch)}")

        # ----- Encoder stats -----
        encoder_mask_bool = (encoder_mask != pad_idx)
        batch_enc_lens = encoder_mask_bool.sum(dim=1).tolist()
        enc_lengths.extend(batch_enc_lens)

        enc_valid_tokens = encoder_ids[encoder_mask_bool]
        enc_vocab.update(map(int, enc_valid_tokens.tolist()))

        # ----- Decoder stats (if targets exist) -----
        if decoder_targets is not None:
            dec_mask = (decoder_targets != pad_idx)
            batch_dec_lens = dec_mask.sum(dim=1).tolist()
            dec_lengths.extend(batch_dec_lens)

            dec_valid_tokens = decoder_targets[dec_mask]
            dec_vocab.update(map(int, dec_valid_tokens.tolist()))

    stats = {
        "mean_encoder_length": float(np.mean(enc_lengths)),
        "encoder_vocab_size": len(enc_vocab),
        "mean_decoder_length": float(np.mean(dec_lengths)) if dec_lengths else None,
        "decoder_vocab_size": len(dec_vocab),
    }

    return stats

if __name__ == "__main__":
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    train_before = compute_stats_before("train", tokenizer)
    dev_before   = compute_stats_before("dev", tokenizer)

    print("=== BEFORE PREPROCESSING (Table 1) ===")
    print("Train:", train_before)
    print("Dev:  ", dev_before)

    train_loader, dev_loader, test_loader = load_t5_data(batch_size=32, test_batch_size=32)

    train_after = collect_stats_after(train_loader)
    dev_after   = collect_stats_after(dev_loader)

    print("\n=== AFTER PREPROCESSING (Table 2) ===")
    print("Train:", train_after)
    print("Dev:  ", dev_after)