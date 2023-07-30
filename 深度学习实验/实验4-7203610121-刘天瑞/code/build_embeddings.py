# -*- coding: gbk -*-
from pathlib import Path
import numpy as np
import os

path = 'Data/Online_Shopping'

if __name__ == '__main__':
    with open(os.path.join(path, 'vocab.words.txt'), 'r', encoding = 'utf-8') as f:
        word_to_idx = {line.strip(): idx + 1 for idx, line in enumerate(f)}
    with open(os.path.join(path, 'vocab.words.txt'), 'r', encoding = 'utf-8') as f:
        word_to_found = {line.strip(): False for line in f}

    # ???????
    size_vocab = len(word_to_idx)

    # ?????embedding????
    embeddings = np.zeros((size_vocab + 1, 300))

    found = 0
    print('Reading From W2V File (may take a while,please be patient and wait!)')
    with Path('sgns.merge.word').open(encoding = 'utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('-- At Line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if (word in word_to_idx) and (not word_to_found[word]):
                word_to_found[word] = True
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('-- Have done. Found {} vectors for {} words'.format(found, size_vocab))
    print('-- The first place is left for <UNK> and <PAD>')

    # ???? np.array
    np.savez_compressed('Data/Online_Shopping/w2v.npz', embeddings = embeddings)



