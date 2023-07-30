# -*- coding: gbk -*-
import os
import json
import jieba
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from custom_dataset import Online_Shopping

def Online_Shopping_Loader(path = "Data/Online_Shopping_10_cats.csv", save = True, batch_size = 64, shuffle = True):
    """��ȡonline shopping���ݼ�"""
    # ��ȡcsv
    csv = pd.read_csv(path, low_memory = False)  # ��ֹ��������
    csv_df = pd.DataFrame(csv)
    del csv_df["label"]

    # �����ǩ
    data_class = {"�鼮": 0, "ƽ��": 1, "�ֻ�": 2, "ˮ��": 3, "ϴ��ˮ": 4, "��ˮ��": 5, "��ţ": 6, "�·�": 7, "�����": 8, "�Ƶ�": 9}
    csv_df["cat"] = csv_df["cat"].map(data_class)

    # ����ǩ����Ϊjson�ļ�
    if save:
        json_file = json.dumps(dict((v, k) for k, v in data_class.items()), indent = 4)
        with open('Data/Online_Shopping.json', 'w', encoding = "utf-8") as f:
            f.write(json_file)

    # �з����ݼ�
    train_text = []
    train_label = []
    test_text = []
    test_label = []
    ver_text = []
    ver_label = []

    # ͳ�ƴ�Ƶ����ѵ���������ǩ��
    count = 0
    vocab_dict = {}
    label_dict = {}
    for _, row in csv_df.iterrows():
        count += 1
        # ���ı����ִʴ���
        label = row['cat']
        text = row['review']
        split_row = jieba.lcut(text)
        if count % 5 == 0:  # ���Լ�
            test_text.append(split_row)
            test_label.append(label)
        elif count % 5 == 4:  # ��֤��
            ver_text.append(split_row)
            ver_label.append(label)
        else:  # ѵ����
            train_text.append(split_row)
            train_label.append(label)
            # �����Ƶ���ǩ��ͳ��
            for word in split_row:
                if word == '\ufeff' or '':
                    pass
                elif word in vocab_dict:
                    vocab_dict[word] += 1
                else:
                    vocab_dict[word] = 1
            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1

    # �����ݷ���dataloader
    train_dataset = Online_Shopping(train_text, train_label)
    test_dataset = Online_Shopping(test_text, test_label)
    ver_dataset = Online_Shopping(ver_text, ver_label)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle)
    ver_loader = DataLoader(ver_dataset, batch_size = batch_size, shuffle = shuffle)

    return train_loader, test_loader, ver_loader


def Jena_loader(path="Data/Online_Shopping_10_cats.csv", save = True, batch_size = 64, shuffle = True):
    """��ȡjena_climate���ݼ�"""

