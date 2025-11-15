import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_data_path = os.path.join(data_folder, f"{split}.nl")
        nl_data = load_lines(nl_data_path)

        if split != "test":
            sql_data_path = os.path.join(data_folder, f"{split}.sql")
            sql_data = load_lines(sql_data_path)

        examples = []

        for idx, nl_text in enumerate(nl_data):
            
            encoder_tokens = tokenizer.encode(
                nl_text,
                add_special_tokens=True, 
                truncation=True,
                max_length=512,
            )
            encoder_tensor = torch.tensor(encoder_tokens, dtype=torch.long)

            example = {
                "encoder_input_ids": encoder_tensor,
            }

            if split != "test":
                sql_text = sql_data[idx]
                decoder_tokens = tokenizer.encode(
                    sql_text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=512,
                )
                decoder_tensor = torch.tensor(decoder_tokens, dtype=torch.long)
                example["decoder_target_ids"] = decoder_tensor

            examples.append(example)
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [item['encoder_input_ids'] for item in batch]
    decoder_ids_list = [item['decoder_target_ids'] for item in batch]

    encoder_ids = pad_sequence(
        encoder_ids_list, batch_first=True, padding_value=PAD_IDX
    ) 
    encoder_mask = (encoder_ids != PAD_IDX).long()

    decoder_targets = pad_sequence(
        decoder_ids_list, batch_first=True, padding_value=PAD_IDX
    )
    decoder_inputs = torch.full_like(decoder_targets, PAD_IDX)
    decoder_inputs[:, 1:] = decoder_targets[:, :-1]
    initial_decoder_inputs = decoder_inputs[:, 0].unsqueeze(1)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [item['encoder_input_ids'] for item in batch]
    encoder_ids = pad_sequence(
        encoder_ids_list, batch_first=True, padding_value=PAD_IDX
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full(
        (encoder_ids.size(0), 1),
        PAD_IDX,
        dtype=torch.long,
    )
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x