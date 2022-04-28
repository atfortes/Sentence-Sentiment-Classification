import torch
import warnings
import torch.nn as nn
from test import test
from train import train
import torch.optim as optim
from model import TextClassifier
from torchtext import data, datasets
from torchtext.vocab import Vectors


EPOCHS = 10
PATIENCE = 2
DROPOUT = 0.2
BATCH_SIZE = 64
HIDDEN_SIZE = 256
HIDDEN_LAYERS = 2
PRETRAINED = True
DATA_DIR = '.data/sst/trees'
IN_FILES = ['train.txt', 'dev.txt', 'test.txt']
CSV_FILES = ['train.csv', 'dev.csv', 'test.csv']
warnings.filterwarnings('ignore', category=UserWarning)  # torchtext deprecation warnings


def process_SST():
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False, dtype=torch.long)

    train_set, val_set, test_set = datasets.SST.splits(
        TEXT, LABEL, fine_grained=True, train_subtrees=False
    )

    print('train_set size:\t', len(train_set))
    print('val_set size:\t', len(val_set))
    print('test_set size:\t', len(test_set))
    
    TEXT.build_vocab(train_set, vectors=Vectors(name='vector.txt', cache='./word-embeddings'))
    LABEL.build_vocab(train_set)

    print('vocab size:\t', len(TEXT.vocab))
    print('embedding dim:\t', TEXT.vocab.vectors.size()[1])

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_set, val_set, test_set), batch_size=BATCH_SIZE
    )

    return TEXT, LABEL, train_iter, val_iter, test_iter


if __name__ == '__main__':
    print('Processing SST Dataset...')
    TEXT, LABEL, train_iter, val_iter, test_iter = process_SST()
    
    print('Creating TextClassifier Model...')
    model = TextClassifier(HIDDEN_LAYERS, HIDDEN_SIZE, DROPOUT, TEXT, LABEL, PRETRAINED)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    print('Training...')
    loss_hist, accu_hist = train(
        model, train_iter, val_iter, criterion, optimizer, EPOCHS, BATCH_SIZE, PATIENCE
    )

    print('Testing...')
    accu = test(model, test_iter, BATCH_SIZE)
    print(f'test_acc={accu:.4f}')
