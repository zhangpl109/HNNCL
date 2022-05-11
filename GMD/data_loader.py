from pathlib import Path

import numpy as np
import torch
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfTransformer
from torch.utils.data import TensorDataset, DataLoader


class Vocabulary(object):
    def __init__(self, path):
        with open(path) as f:
            self.id2word = [x.strip() for x in f]
        word2id = {x: i for i, x in enumerate(self.id2word)}
        assert len(word2id) == len(self.id2word)

    def __len__(self):
        return len(self.id2word)


def load_train_data(path):
    """
    Returns: tfidf with size (n_documents, vocab_size).
    """
    bow = load_npz(path)  # bow[i][j] is the number of word j in document i.
    tfidf_transformer = TfidfTransformer(norm='l1')
    tfidf = tfidf_transformer.fit_transform(bow)
    tfidf = tfidf.astype(np.float32).toarray()
    tfidf = torch.tensor(tfidf)
    return tfidf


def load_word_vectors(path):
    with open(path) as f:
        # line i (0-based indexing) is the word vector of word i
        wv = torch.tensor([[float(y) for y in x.split()] for x in f], dtype=torch.float32)
    return wv


def load_data(data_dir, batch_size, use_word_vectors):
    data_dir = Path(data_dir)
    vocab = Vocabulary(data_dir / 'vocab.txt')

    data = load_train_data(data_dir / 'train.bow.npz')
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    word_vec = load_word_vectors(data_dir / 'glove.300.txt') if use_word_vectors else None

    return vocab, data_loader, word_vec
