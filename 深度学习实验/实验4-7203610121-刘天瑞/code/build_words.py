# -*- coding: gbk -*-
import os
import json
import jieba
import pandas as pd

def build_words(path="Data/Online_Shopping/online_shopping_10_cats.csv", save_json = True,
                save_path="Data/Online_Shopping"):
    """��ȡonline shopping���ݼ�"""
    # ��ȡcsv
    csv = pd.read_csv(path, low_memory = False, encoding = 'utf-8-sig')  # ��ֹ�������������\ufeff
    csv_df = pd.DataFrame(csv)
    del csv_df["label"]

    # �����ǩ
    data_class = {"�鼮": 0, "ƽ��": 1, "�ֻ�": 2, "ˮ��": 3, "ϴ��ˮ": 4, "��ˮ��": 5, "��ţ": 6, "�·�": 7, "�����": 8, "�Ƶ�": 9}
    csv_df["cat"] = csv_df["cat"].map(data_class)

    # ����ǩ����Ϊjson�ļ�
    if save_json:
        json_file = json.dumps(dict((v, k) for k, v in data_class.items()), indent = 4)
        with open('Data/Online_Shopping/Labels.json', 'w', encoding = "utf-8") as f:
            f.write(json_file)

    # �з����ݼ�
    train_text = []
    train_label = []
    test_text = []
    test_label = []
    ver_text = []
    ver_label = []
    count = 0

    # ͳ�Ʋ���
    for _, row in csv_df.iterrows():
        count += 1
        # ���ı����ִʴ���
        label = row['cat']
        text = str(row['review'])
        if text.strip() != '':
            split_row = [' '.join(jieba.cut(text.strip()))]
        if count % 5 == 0:  # ���Լ�
            test_text.append(split_row)
            test_label.append(label)
        elif count % 5 == 4:  # ��֤��
            ver_text.append(split_row)
            ver_label.append(label)
        else:  # ѵ����
            train_text.append(split_row)
            train_label.append(label)

    # ������
    save_results(train_text, train_label, mode = 'train')
    save_results(test_text, test_label, mode = 'test')
    save_results(ver_text, ver_label, mode = 'ver')

    return


def save_results(text, label, mode, path = "Data/Online_Shopping"):
    with open(os.path.join(path, '{}.words.txt'.format(mode)), 'w', encoding = 'utf-8') as w:
        for i in range(len(text)):
            w.write('{},{}\n'.format(label[i], ''.join(text[i])))


if __name__ == '__main__':
    build_words(path = "Data/Online_Shopping/online_shopping_10_cats.csv")


