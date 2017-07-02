import sys
import numpy as np

vocab = {}

def load_data(filename):
    global vocab, n_vocab
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)   # 単語をIDに変換
        dataset[i] = vocab[word]       # datasetに単語IDを追加
    return dataset

train_data = load_data('ptb.train.txt')
valid_data = load_data('ptb.valid.txt')
test_data = load_data('ptb.test.txt')

print(train_data)
print(vocab)
print('#vocab =', len(vocab))
