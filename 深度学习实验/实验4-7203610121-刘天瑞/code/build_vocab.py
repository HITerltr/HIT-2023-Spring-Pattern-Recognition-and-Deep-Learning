# -*- coding: gbk -*-
import os
from collections import Counter

MIN_COUNT = 1

path = 'Data/Online_Shopping'


def words(name):
    return '{}.words.txt'.format(name)


if __name__ == '__main__':
    print('Build vocab words')
    counter_words = Counter()
    with open(os.path.join(path, words('train')), encoding = 'utf-8') as f:
        for line in f:
            label = line[:1]
            line = line[2:]
            counter_words.update(line.strip().split())
    vocab_words = {w for w, c in counter_words.items() if c >= MIN_COUNT}

    with open(os.path.join(path, 'vocab.words.txt'), 'w', encoding = 'utf-8') as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('Have done. Kept {} out of {}'.format(len(vocab_words), len(counter_words)))