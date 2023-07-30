# -*- coding: gbk -*-
import os
from collections import Counter
from custom_dataset import Online_Shopping
from torch.utils.data import DataLoader


def words(name):
    return '{}.words.txt'.format(name)


def Load_Online_Shopping(path="Data/Online_Shopping", max_len = 100, batch_size = 64, shuffle = True):
    print('Build online shopping dataset!')

    dataloaders = []

    with open(os.path.join(path, 'vocab.words.txt'), 'r', encoding = 'utf-8') as f:
        word_to_idx = {line.strip(): idx + 1 for idx, line in enumerate(f)}

    for n in ['train', 'ver', 'test']:
        full_text = []
        full_label = []
        with open(os.path.join(path, words(n)), encoding = 'utf-8') as f:
            for line in f:
                label = line[:1]
                text = line[2:]
                text_list = text.strip().split()
                text_idx_list = [0] * max_len
                for i, word in enumerate(text_list):
                    if word in word_to_idx:
                        word_idx = word_to_idx[word]
                    else:
                        word_idx = 0
                    if i < max_len:
                        text_idx_list[i] = word_idx
                    else:
                        break
                full_text.append(list(map(int, text_idx_list)))  # �б�����ת������֤ȫΪint
                full_label.append(int(label))

        # �Ѿ��ռ��������ȫ�����ݣ�ת��dataset
        dataset = Online_Shopping(full_text, full_label)
        loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, drop_last = True)
        dataloaders.append(loader)
        print('-- {} dataloader have done!'.format(n))

    return dataloaders




