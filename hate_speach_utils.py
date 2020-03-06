import torch
import torchtext
from torchtext import data
from torchtext.vocab import Vectors, GloVe
import os
from tqdm import tqdm
from datetime import datetime
import logging
import pprint
import io
from transformers import RobertaTokenizer
from tokenizers import CharBPETokenizer
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

def get_average_embedding(vector_map):
    """
    From the dictionary of embeddings gets the average out.
    """
    embeds = torch.cat(list(map(lambda x: x.view(1, -1), vector_map.values())), 0)
    return torch.mean(embeds, 0)

def load_vectors(fname, train_vocab, device):
    """
    Modified from https://fasttext.cc/docs/en/english-vectors.html.
    This loads fasttext vectors for words that have been encountered in the
    vocabulary `train_vocab`.
    We also build a string to inter map to get inter index for the words.
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    stoi = {}
    loaded_vectors = 0
    words = {word.lower().strip() for word in train_vocab.freqs}
    for idx, line in tqdm(enumerate(fin)):
        tokens = line.rstrip().split(' ')
        if tokens[0] in words:
            stoi[tokens[0]] = idx
            data[idx] = torch.tensor(list(map(float, tokens[1:])), device=device)
            loaded_vectors += 1
    logger = logging.getLogger('logger')
    logger.info('Number of vectors loaded from fasttext {}'.format(loaded_vectors))
    return data, stoi

def torchtext_iterators(args):
    """
    Builds torchtext iterators from the files.
    """
    logger = logging.getLogger('logger')
    logger.info('Starting to load data and create iterators.')

    # Tokenizer.
    if args['model_name'] == 'roberta':
        tokenizer = lambda x : [x]
    elif args['subword']:
        tokenizer = 'subword'
    elif args['bpe']:
        bpe_tokenizer = CharBPETokenizer('log/bpe-trained-vocab.json', 'log/bpe-trained-merges.txt')
        tokenizer = lambda x : bpe_tokenizer.encode(x).tokens
    else:
        tokenizer = None

    # `sequential` does not tokenize the label.
    label = data.Field(batch_first=True, sequential=False)
    text = data.Field(batch_first=True, lower=True, tokenize=tokenizer)

    fields = [('text', text), ('label', label)]
    train = data.TabularDataset(args['train_path'], 'tsv', fields, skip_header=True)
    valid = data.TabularDataset(args['valid_path'], 'tsv', fields, skip_header=True)
    test = data.TabularDataset(args['test_path'], 'tsv', [('text', text)], skip_header=True)

    text.build_vocab(train, min_freq=args['min_freq'])
    label.build_vocab(train)

    train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
            (train, valid, test), batch_size=args['batch_size'], repeat=False,
            device=torch.device(args['device']), sort=False,
            sort_within_batch=False)

    if not args['no_pretrained_vectors']:
        if not args['load_vectors_manually']:
            logger.info('Starting to load vectors from Glove.')
            text.vocab.load_vectors(vectors=GloVe(name='6B'))
        else:
            logger.info('Starting to manually load vectors from FastText.')
            vector_map, stoi = load_vectors(args['fasttext_path'], text.vocab, torch.device(args['device']))
            average_embed = get_average_embedding(vector_map)
            text.vocab.set_vectors(stoi, vector_map, 300, unk_init=lambda x: average_embed.clone())
            text.vocab.vectors[text.vocab.stoi['<unk>']] = average_embed.clone()

    logger.info('Built train vocabulary of {} words'.format(len(text.vocab)))
    return train_iter, valid_iter, test_iter, text, label

def predict_write_to_file(module, val_iter, args):
    mode = module.training
    module.eval()
    predictions = []

    for batch in tqdm(val_iter):
        scores = module.forward(batch.text)
        preds = scores.argmax(1).squeeze()
        predictions += list(preds.cpu().numpy())

    # Write predictions to file.
    with open(os.path.join('log', args['checkpoint_path'], 'test_results.txt'), 'w') as out_file:
        for pred in predictions:
            out_file.write(args['LABEL'].vocab.itos[pred] + '\n')

    module.train(mode)
