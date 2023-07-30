# -*- coding: gbk -*-
import os
import json
import jieba
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from custom_dataset import Online_Shopping

def Online_Shopping_Loader(path = "Data/Online_Shopping_10_cats.csv", save = True, batch_size = 64, shuffle = True):
    """读取online shopping数据集"""
    # 读取csv
    csv = pd.read_csv(path, low_memory = False)  # 防止弹出警告
    csv_df = pd.DataFrame(csv)
    del csv_df["label"]

    # 处理标签
    data_class = {"书籍": 0, "平板": 1, "手机": 2, "水果": 3, "洗发水": 4, "热水器": 5, "蒙牛": 6, "衣服": 7, "计算机": 8, "酒店": 9}
    csv_df["cat"] = csv_df["cat"].map(data_class)

    # 将标签保存为json文件
    if save:
        json_file = json.dumps(dict((v, k) for k, v in data_class.items()), indent = 4)
        with open('Data/Online_Shopping.json', 'w', encoding = "utf-8") as f:
            f.write(json_file)

    # 切分数据集
    train_text = []
    train_label = []
    test_text = []
    test_label = []
    ver_text = []
    ver_label = []

    # 统计词频（仅训练集）与标签数
    count = 0
    vocab_dict = {}
    label_dict = {}
    for _, row in csv_df.iterrows():
        count += 1
        # 对文本做分词处理
        label = row['cat']
        text = row['review']
        split_row = jieba.lcut(text)
        if count % 5 == 0:  # 测试集
            test_text.append(split_row)
            test_label.append(label)
        elif count % 5 == 4:  # 验证集
            ver_text.append(split_row)
            ver_label.append(label)
        else:  # 训练集
            train_text.append(split_row)
            train_label.append(label)
            # 加入词频与标签数统计
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

    # 将数据放入dataloader
    train_dataset = Online_Shopping(train_text, train_label)
    test_dataset = Online_Shopping(test_text, test_label)
    ver_dataset = Online_Shopping(ver_text, ver_label)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle)
    ver_loader = DataLoader(ver_dataset, batch_size = batch_size, shuffle = shuffle)

    return train_loader, test_loader, ver_loader


def Jena_loader(path="Data/Online_Shopping_10_cats.csv", save = True, batch_size = 64, shuffle = True):
    """读取jena_climate数据集"""

